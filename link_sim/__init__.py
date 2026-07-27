"""Link Simulator Package

Software-in-the-loop CCSDS link simulation with deterministic fault injection.
Provides transport abstraction, fault modeling, and self-contained replay artifacts.
"""

__version__ = "0.1.0"

from .transport import Direction, Transport, InMemoryTransport, UdpTransport, SidebandEnvelope, TransportFrame
from .fault_model import FaultProfile, FaultDecision
from .virtual_clock import VirtualClock, SimulationTime
from .link_simulator import LinkSimulator
from .control import ControlType, LinkControl, LinkControlMessage, LinkControlType
from .mission_link import MissionLink

__all__ = [
    "Transport",
    "Direction",
    "InMemoryTransport",
    "UdpTransport",
    "SidebandEnvelope",
    "TransportFrame",
    "FaultProfile",
    "FaultDecision",
    "VirtualClock",
    "SimulationTime",
    "LinkSimulator",
    "ControlType",
    "LinkControl",
    "LinkControlMessage",
    "LinkControlType",
    "MissionLink",
]
