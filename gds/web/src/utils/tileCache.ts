export interface TileCacheOptions {
  maxEntries?: number;
  maxBytes?: number;
  maxConcurrent?: number;
}

interface Entry {
  url: string;
  bytes: number;
  lastUsed: number;
}

interface Pending {
  key: string;
  resolve: (value: string) => void;
  reject: (reason: unknown) => void;
}

/** Bounded, cancelable browser tile cache. It owns object URLs and revokes them on eviction. */
export class TileCache {
  readonly maxEntries: number;
  readonly maxBytes: number;
  readonly maxConcurrent: number;
  private readonly entries = new Map<string, Entry>();
  private readonly pending = new Map<string, Pending[]>();
  private readonly queue: string[] = [];
  private active = 0;
  private bytes = 0;

  constructor(options: TileCacheOptions = {}) {
    this.maxEntries = options.maxEntries ?? 128;
    this.maxBytes = options.maxBytes ?? 32 * 1024 * 1024;
    this.maxConcurrent = options.maxConcurrent ?? 8;
  }

  get size(): number {
    return this.entries.size;
  }

  get usedBytes(): number {
    return this.bytes;
  }

  get(key: string): string | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    entry.lastUsed = Date.now();
    return entry.url;
  }

  async load(key: string, signal?: AbortSignal): Promise<string> {
    const cached = this.get(key);
    if (cached) return cached;
    if (signal?.aborted) throw new DOMException("Tile request aborted", "AbortError");
    return new Promise<string>((resolve, reject) => {
      const waiters = this.pending.get(key) ?? [];
      waiters.push({ key, resolve, reject });
      this.pending.set(key, waiters);
      if (waiters.length === 1) this.queue.push(key);
      if (signal) {
        signal.addEventListener(
          "abort",
          () => {
            const current = this.pending.get(key) ?? [];
            const index = current.findIndex((item) => item.resolve === resolve);
            if (index >= 0) current.splice(index, 1);
            if (current.length === 0) this.pending.delete(key);
            reject(new DOMException("Tile request aborted", "AbortError"));
          },
          { once: true },
        );
      }
      this.pump();
    });
  }

  clear(): void {
    for (const entry of this.entries.values()) URL.revokeObjectURL(entry.url);
    this.entries.clear();
    this.bytes = 0;
  }

  cancelOutside(keys: Set<string>): void {
    for (const key of this.pending.keys()) {
      if (!keys.has(key)) {
        for (const waiter of this.pending.get(key) ?? []) {
          waiter.reject(new DOMException("Tile request canceled", "AbortError"));
        }
        this.pending.delete(key);
      }
    }
  }

  private pump(): void {
    while (this.active < this.maxConcurrent && this.queue.length > 0) {
      const key = this.queue.shift();
      if (!key || !this.pending.has(key)) continue;
      this.active += 1;
      void this.fetchOne(key).finally(() => {
        this.active -= 1;
        this.pump();
      });
    }
  }

  private async fetchOne(key: string): Promise<void> {
    const waiters = this.pending.get(key) ?? [];
    try {
      const response = await fetch(key, { credentials: "same-origin" });
      if (!response.ok) throw new Error(`Tile request failed (${response.status})`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      this.entries.set(key, { url, bytes: blob.size, lastUsed: Date.now() });
      this.bytes += blob.size;
      this.evict();
      for (const waiter of waiters) waiter.resolve(url);
    } catch (error) {
      for (const waiter of waiters) waiter.reject(error);
    } finally {
      this.pending.delete(key);
    }
  }

  private evict(): void {
    while (this.entries.size > this.maxEntries || this.bytes > this.maxBytes) {
      const oldest = [...this.entries.entries()].sort((a, b) => a[1].lastUsed - b[1].lastUsed)[0];
      if (!oldest) return;
      const [key, entry] = oldest;
      URL.revokeObjectURL(entry.url);
      this.entries.delete(key);
      this.bytes -= entry.bytes;
    }
  }
}
