import { DestroyRef, Injectable, inject, signal } from '@angular/core';

const STORAGE_KEY = 'rv-sidebar-collapsed';

@Injectable({ providedIn: 'root' })
export class LayoutStateService {
  private readonly destroyRef = inject(DestroyRef);

  readonly collapsed = signal<boolean>(this.readInitialCollapsed());
  readonly mobileOpen = signal<boolean>(false);
  readonly isMobile = signal<boolean>(this.checkIsMobile());
  readonly fullBleed = signal<boolean>(false);

  constructor() {
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      const mql = window.matchMedia('(max-width: 1023px)');
      const onChange = (event: MediaQueryListEvent) => {
        this.isMobile.set(event.matches);
        if (!event.matches) {
          this.mobileOpen.set(false);
        }
      };
      this.isMobile.set(mql.matches);
      mql.addEventListener('change', onChange);
      this.destroyRef.onDestroy(() => mql.removeEventListener('change', onChange));
    }
  }

  toggleSidebar(): void {
    if (this.isMobile()) {
      this.mobileOpen.update((v) => !v);
      return;
    }
    this.setCollapsed(!this.collapsed());
  }

  setCollapsed(next: boolean): void {
    this.collapsed.set(next);
    try {
      localStorage.setItem(STORAGE_KEY, String(next));
    } catch {
      // Ignorar em contextos sem localStorage
    }
  }

  closeMobileMenu(): void {
    this.mobileOpen.set(false);
  }

  setFullBleed(enabled: boolean): void {
    this.fullBleed.set(enabled);
  }

  private readInitialCollapsed(): boolean {
    try {
      return localStorage.getItem(STORAGE_KEY) === 'true';
    } catch {
      return false;
    }
  }

  private checkIsMobile(): boolean {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return false;
    }
    return window.matchMedia('(max-width: 1023px)').matches;
  }
}
