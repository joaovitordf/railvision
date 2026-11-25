import { TestBed } from '@angular/core/testing';
import { LayoutStateService } from './layout-state.service';

describe('LayoutStateService', () => {
  beforeEach(() => {
    localStorage.removeItem('rv-sidebar-collapsed');
    TestBed.configureTestingModule({});
  });

  it('toggles desktop collapse and persists preference in localStorage', () => {
    const service = TestBed.inject(LayoutStateService);
    service.isMobile.set(false);
    expect(service.collapsed()).toBe(false);

    service.toggleSidebar();
    expect(service.collapsed()).toBe(true);
    expect(localStorage.getItem('rv-sidebar-collapsed')).toBe('true');

    service.toggleSidebar();
    expect(service.collapsed()).toBe(false);
    expect(localStorage.getItem('rv-sidebar-collapsed')).toBe('false');
  });

  it('toggles overlay drawer in mobile mode without mutating desktop collapsed state', () => {
    const service = TestBed.inject(LayoutStateService);
    service.isMobile.set(true);
    service.setCollapsed(false);

    service.toggleSidebar();
    expect(service.mobileOpen()).toBe(true);
    expect(service.collapsed()).toBe(false);

    service.closeMobileMenu();
    expect(service.mobileOpen()).toBe(false);
  });
});
