import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
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
      [class.rv-shell--collapsed]="sidebarCollapsed()"
      [class.rv-shell--mobile-open]="mobileMenuOpen()"
    >
      <div class="rv-backdrop" (click)="closeMobileMenu()" aria-hidden="true"></div>

      <rv-sidebar
        [collapsed]="sidebarCollapsed()"
        (toggleCollapse)="toggleSidebarCollapse()"
        (navigate)="closeMobileMenu()"
      />

      <div class="rv-main-column">
        <rv-topbar (toggleMobileMenu)="toggleMobileMenu()" />

        <main class="rv-content">
          <router-outlet />
        </main>

        <rv-footer />
      </div>
    </div>
  `,
})
export class ShellComponent {
  protected readonly sidebarCollapsed = signal<boolean>(false);
  protected readonly mobileMenuOpen = signal<boolean>(false);

  protected toggleSidebarCollapse(): void {
    this.sidebarCollapsed.update((value) => !value);
  }

  protected toggleMobileMenu(): void {
    this.mobileMenuOpen.update((value) => !value);
  }

  protected closeMobileMenu(): void {
    this.mobileMenuOpen.set(false);
  }
}
