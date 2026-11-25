import { Component, HostListener, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { LayoutStateService } from '../../services/layout-state.service';
import { FooterComponent } from '../footer/footer.component';
import { SidebarComponent } from '../sidebar/sidebar.component';
import { TopbarComponent } from '../topbar/topbar.component';

@Component({
  selector: 'rv-shell',
  standalone: true,
  imports: [RouterOutlet, SidebarComponent, TopbarComponent, FooterComponent],
  template: `
    <div
      class="rv-shell"
      [class.rv-shell--collapsed]="layout.collapsed() && !layout.isMobile()"
      [class.rv-shell--mobile-open]="layout.mobileOpen()"
      [class.rv-shell--fullbleed]="layout.fullBleed()"
    >
      <div class="rv-backdrop" (click)="layout.closeMobileMenu()" aria-hidden="true"></div>

      <rv-sidebar />

      <div class="rv-main-column">
        <rv-topbar />

        <main class="rv-content">
          <router-outlet />
        </main>

        <rv-footer />
      </div>
    </div>
  `,
})
export class ShellComponent {
  protected readonly layout = inject(LayoutStateService);

  @HostListener('document:keydown.escape')
  protected onEscape(): void {
    if (this.layout.mobileOpen()) {
      this.layout.closeMobileMenu();
    }
  }
}
