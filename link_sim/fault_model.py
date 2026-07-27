"""Fault profile and deterministic fault injection.

Section 9.2: Deterministic fault model using canonical counter PRF.
Each fault decision uses SHA-256 counter mode with fixed seed, run ID, and frame ID.
"""

import hashlib
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional


class FaultStage(IntEnum):
    """Fault injection stages. Section 9.2: fixed stage codes."""
    LOSS = 1
    DUPLICATE = 2
    CORRUPT_DECISION = 3
    CORRUPT_BIT = 4
    JITTER = 5
    REORDER = 6


@dataclass
class FaultProfile:
    """Fault profile configuration for one direction.

    Section 9.2: Rates in PPM (0..1_000_000), deterministic bounded integer mapping.
    """
    schema_version: int = 1
    profile_revision: int = 0

    # Latency and jitter in nanoseconds
    base_latency_ns: int = 0
    jitter_abs_ns: int = 0  # Uniform in [-jitter, +jitter]

    # Loss and duplication rates (0..1_000_000 PPM)
    frame_loss_rate_ppm: int = 0
    frame_duplicate_rate_ppm: int = 0

    # Corruption: rate + bits per corrupted frame
    corrupt_frame_rate_ppm: int = 0
    bits_per_corrupt_frame: int = 0

    # Reordering
    reorder_window_slots: int = 0  # 0 = no reordering
    reorder_slot_ns: int = 0

    # Bandwidth in bits per second
    bitrate_bps: int = 1_000_000_000  # 1 Gbps default (no throttling)

    def validate(self) -> None:
        """Validate profile parameters."""
        if not (0 <= self.frame_loss_rate_ppm <= 1_000_000):
            raise ValueError(f"Invalid loss rate: {self.frame_loss_rate_ppm}")
        if not (0 <= self.frame_duplicate_rate_ppm <= 1_000_000):
            raise ValueError(f"Invalid duplicate rate: {self.frame_duplicate_rate_ppm}")
        if not (0 <= self.corrupt_frame_rate_ppm <= 1_000_000):
            raise ValueError(f"Invalid corrupt rate: {self.corrupt_frame_rate_ppm}")
        if self.bits_per_corrupt_frame < 0:
            raise ValueError(f"Invalid bits_per_corrupt_frame: {self.bits_per_corrupt_frame}")
        if self.base_latency_ns < self.jitter_abs_ns:
            raise ValueError(
                f"base_latency_ns ({self.base_latency_ns}) must be >= jitter_abs_ns ({self.jitter_abs_ns})"
            )
        if self.bitrate_bps <= 0:
            raise ValueError(f"bitrate_bps must be > 0: {self.bitrate_bps}")
        if self.reorder_window_slots < 0:
            raise ValueError(f"Invalid reorder_window_slots: {self.reorder_window_slots}")
        if self.reorder_slot_ns < 0:
            raise ValueError(f"Invalid reorder_slot_ns: {self.reorder_slot_ns}")


@dataclass
class FaultDecision:
    """Result of fault injection for one frame copy."""
    is_lost: bool
    has_duplicate: bool
    is_corrupted: bool
    corrupted_bits: List[int]  # Bit offsets that were flipped
    latency_ns: int
    jitter_ns: int
    reorder_slots: int
    due_ns: int  # ingress_time + base_latency + jitter + reorder_slots * reorder_slot_ns
    tx_start_ns: int  # max(due_ns, link_available)
    tx_duration_ns: int
    release_ns: int  # tx_start + tx_duration


class FaultModel:
    """Deterministic fault injection using counter PRF.

    Section 9.2: Each draw uses SHA-256("link-fault-v1" || seed || run_id ||
    profile_revision || direction || link_frame_id || copy_index || stage_code || draw_index).
    """

    PRF_DOMAIN = b"link-fault-v1"

    def __init__(self, seed: int, simulation_run_id: int):
        """Initialize fault model.

        Args:
            seed: U64 seed for reproducibility
            simulation_run_id: U64 unique run identifier
        """
        self.seed = seed
        self.simulation_run_id = simulation_run_id

    def _prf(
        self,
        profile_revision: int,
        direction: int,
        link_frame_id: int,
        copy_index: int,
        stage_code: int,
        draw_index: int,
    ) -> int:
        """Counter PRF: SHA-256 hash -> U64.

        Returns first 8 bytes of hash as big-endian U64.
        """
        # Pack: domain || seed || run_id || revision || direction || frame_id || copy || stage || draw
        data = struct.pack(
            ">8sQQIBQHBI",
            self.PRF_DOMAIN,
            self.seed,
            self.simulation_run_id,
            profile_revision,
            direction,
            link_frame_id,
            copy_index,
            stage_code,
            draw_index,
        )
        digest = hashlib.sha256(data).digest()
        return struct.unpack(">Q", digest[:8])[0]

    def _draw_bool(
        self,
        rate_ppm: int,
        profile_revision: int,
        direction: int,
        link_frame_id: int,
        copy_index: int,
        stage_code: int,
        draw_index: int,
    ) -> bool:
        """Draw boolean decision based on rate in PPM.

        rate_ppm=0 -> always False
        rate_ppm=1_000_000 -> always True
        rate_ppm in (0, 1_000_000) -> True if draw < threshold

        Threshold = floor(rate_ppm * 2^64 / 1_000_000) computed with U128.
        """
        if rate_ppm == 0:
            return False
        if rate_ppm == 1_000_000:
            return True

        draw = self._prf(profile_revision, direction, link_frame_id, copy_index, stage_code, draw_index)

        # Compute threshold = floor(rate_ppm * 2^64 / 1_000_000) using Python big-int
        threshold = (rate_ppm * (2**64)) // 1_000_000

        return draw < threshold

    def _draw_bounded(
        self,
        upper_bound: int,
        profile_revision: int,
        direction: int,
        link_frame_id: int,
        copy_index: int,
        stage_code: int,
        draw_index: int,
    ) -> int:
        """Draw bounded integer in [0, upper_bound).

        Uses high-half multiplication: floor(draw_u64 * N / 2^64).
        """
        if upper_bound <= 0:
            raise ValueError(f"upper_bound must be > 0: {upper_bound}")

        draw = self._prf(profile_revision, direction, link_frame_id, copy_index, stage_code, draw_index)

        # Compute floor(draw * upper_bound / 2^64) using Python big-int
        result = (draw * upper_bound) // (2**64)
        return result

    def apply_faults(
        self,
        profile: FaultProfile,
        direction: int,
        link_frame_id: int,
        frame_bits: int,
        ingress_time_ns: int,
        link_available_ns: int,
        *,
        copy_index: int = 0,
        include_duplicate: bool = True,
    ) -> FaultDecision:
        """Apply fault profile to one frame copy.

        Section 9.2 order:
        1. Loss
        2. Duplicate
        3. Corruption (per copy)
        4. Latency/jitter (per copy)
        5. Reorder scheduling (per copy)
        6. Bandwidth serialization

        Args:
            profile: Fault profile
            direction: 0=uplink, 1=downlink
            link_frame_id: Frame ID assigned by Link Simulator
            frame_bits: Frame size in bits (for corruption and bandwidth)
            ingress_time_ns: Ingress admission time
            link_available_ns: Link serializer available time
            copy_index: Independent PRF copy index. The original is 0 and a
                duplicate is evaluated as copy 1, rather than reusing the
                original decision.
            include_duplicate: Only the original copy may create another copy.

        Returns:
            FaultDecision for the requested copy
        """
        profile.validate()
        if isinstance(copy_index, bool) or not 0 <= copy_index <= 0xFFFF:
            raise ValueError("copy_index must fit U16")
        rev = profile.profile_revision

        # Stage 1: Loss (draw_index=0)
        is_lost = self._draw_bool(
            profile.frame_loss_rate_ppm, rev, direction, link_frame_id, copy_index, FaultStage.LOSS, 0
        )

        # Stage 2: Duplicate (draw_index=0)
        has_duplicate = include_duplicate and self._draw_bool(
            profile.frame_duplicate_rate_ppm,
            rev,
            direction,
            link_frame_id,
            copy_index,
            FaultStage.DUPLICATE,
            0,
        )

        # Stage 3: Corruption decision (draw_index=0)
        is_corrupted = self._draw_bool(
            profile.corrupt_frame_rate_ppm, rev, direction, link_frame_id, copy_index, FaultStage.CORRUPT_DECISION, 0
        )

        corrupted_bits: List[int] = []
        if is_corrupted and profile.bits_per_corrupt_frame > 0:
            # Validate bits_per_corrupt_frame
            if profile.bits_per_corrupt_frame > frame_bits:
                raise ValueError(
                    f"bits_per_corrupt_frame ({profile.bits_per_corrupt_frame}) > frame_bits ({frame_bits})"
                )

            # Stage 4: Select unique bit offsets
            draw_index = 0
            while len(corrupted_bits) < profile.bits_per_corrupt_frame:
                bit_offset = self._draw_bounded(
                    frame_bits, rev, direction, link_frame_id, copy_index, FaultStage.CORRUPT_BIT, draw_index
                )
                if bit_offset not in corrupted_bits:
                    corrupted_bits.append(bit_offset)
                draw_index += 1

        # Stage 5: Jitter (draw_index=0)
        # signed_jitter = bounded(2*jitter_abs+1) - jitter_abs
        if profile.jitter_abs_ns > 0:
            jitter_range = 2 * profile.jitter_abs_ns + 1
            jitter_unsigned = self._draw_bounded(
                jitter_range, rev, direction, link_frame_id, copy_index, FaultStage.JITTER, 0
            )
            jitter_ns = jitter_unsigned - profile.jitter_abs_ns
        else:
            jitter_ns = 0

        # Stage 6: Reorder (draw_index=0)
        if profile.reorder_window_slots > 0:
            reorder_slots = self._draw_bounded(
                profile.reorder_window_slots + 1, rev, direction, link_frame_id, copy_index, FaultStage.REORDER, 0
            )
        else:
            reorder_slots = 0

        # Compute due time with overflow check
        latency_ns = profile.base_latency_ns
        due_ns = ingress_time_ns + latency_ns + jitter_ns + reorder_slots * profile.reorder_slot_ns

        # Check for overflow (U64 max)
        if due_ns < 0 or due_ns > (2**64 - 1):
            raise OverflowError(f"due_ns overflow: {due_ns}")

        # Bandwidth serialization
        tx_start_ns = max(due_ns, link_available_ns)

        # tx_duration = ceil(frame_bits * 1e9 / bitrate_bps) using U128
        # Python: (frame_bits * 10**9 + bitrate_bps - 1) // bitrate_bps
        tx_duration_ns = (frame_bits * 10**9 + profile.bitrate_bps - 1) // profile.bitrate_bps

        release_ns = tx_start_ns + tx_duration_ns

        # Check for overflow
        if release_ns < 0 or release_ns > (2**64 - 1):
            raise OverflowError(f"release_ns overflow: {release_ns}")

        return FaultDecision(
            is_lost=is_lost,
            has_duplicate=has_duplicate,
            is_corrupted=is_corrupted,
            corrupted_bits=sorted(corrupted_bits),
            latency_ns=latency_ns,
            jitter_ns=jitter_ns,
            reorder_slots=reorder_slots,
            due_ns=due_ns,
            tx_start_ns=tx_start_ns,
            tx_duration_ns=tx_duration_ns,
            release_ns=release_ns,
        )

    def corrupt_frame(self, frame_bytes: bytes, bit_offsets: List[int]) -> bytes:
        """Apply bit corruption to frame.

        Bit offset 0 is MSB (0x80) of byte 0.
        Corruption uses XOR with mask 1 << (7 - (offset mod 8)).
        """
        frame_array = bytearray(frame_bytes)
        frame_bits = len(frame_bytes) * 8

        for bit_offset in bit_offsets:
            if bit_offset < 0 or bit_offset >= frame_bits:
                raise ValueError(f"Bit offset {bit_offset} out of range [0, {frame_bits})")

            byte_index = bit_offset // 8
            bit_index = bit_offset % 8
            mask = 1 << (7 - bit_index)
            frame_array[byte_index] ^= mask

        return bytes(frame_array)
