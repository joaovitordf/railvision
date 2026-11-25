import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { NAVIGATION_ITEMS } from '../../navigation/navigation.config';
import { LayoutStateService } from '../../services/layout-state.service';
import { RuntimeStatusService } from '../../services/runtime-status.service';
import { ShellComponent } from './shell.component';

describe('ShellComponent & Navigation', () => {
  beforeEach(async () => {
    localStorage.removeItem('rv-sidebar-collapsed');
    await TestBed.configureTestingModule({
      imports: [ShellComponent],
      providers: [
        provideRouter([]),
        {
          provide: RuntimeStatusService,
          useValue: {
            getStatus: () => of({ available: true, availableRuntimes: [{ id: 'pytorch_cuda', available: true }] }),
          },
        },
      ],
    }).compileComponents();
  });

  it('renders exactly four primary navigation areas in the sidebar', () => {
    expect(NAVIGATION_ITEMS.map((i) => i.route)).toEqual([
      '/overview',
      '/inspections',
      '/annotations',
      '/lab',
    ]);
    const fixture = TestBed.createComponent(ShellComponent);
    fixture.detectChanges();
    const navLinks = fixture.nativeElement.querySelectorAll('.rv-sidebar__nav a.rv-nav-item');
    expect(navLinks.length).toBe(4);
  });

  it('keeps topbar toggle and footer toggle synchronized in both expanded and collapsed states', () => {
    const fixture = TestBed.createComponent(ShellComponent);
    const layout = TestBed.inject(LayoutStateService);
    layout.isMobile.set(false);
    fixture.detectChanges();

    const shellEl: HTMLElement = fixture.nativeElement.querySelector('.rv-shell');
    const topbarBtn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="topbar-menu-btn"]');
    const footerBtn: HTMLButtonElement = fixture.nativeElement.querySelector('[data-testid="sidebar-footer-toggle"]');

    expect(shellEl.classList.contains('rv-shell--collapsed')).toBe(false);
    topbarBtn.click();
    fixture.detectChanges();
    expect(shellEl.classList.contains('rv-shell--collapsed')).toBe(true);

    // Expanding back via sidebar button and collapsing again works repeatedly
    footerBtn.click();
    fixture.detectChanges();
    expect(shellEl.classList.contains('rv-shell--collapsed')).toBe(false);

    topbarBtn.click();
    fixture.detectChanges();
    expect(shellEl.classList.contains('rv-shell--collapsed')).toBe(true);
  });
});
