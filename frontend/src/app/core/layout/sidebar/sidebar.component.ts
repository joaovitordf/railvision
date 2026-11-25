import { Component, inject } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { NAVIGATION_ITEMS } from '../../navigation/navigation.config';
import { LayoutStateService } from '../../services/layout-state.service';

@Component({
  selector: 'rv-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, MatIconModule, MatTooltipModule],
  template: `
    <aside class="rv-sidebar" aria-label="Navegação principal">
      <div class="rv-sidebar__header">
        <a
          routerLink="/overview"
          class="rv-sidebar__brand"
          (click)="layout.closeMobileMenu()"
          [matTooltip]="isDesktopCollapsed() ? 'RailVision AI' : ''"
          matTooltipPosition="right"
        >
          <span class="rv-sidebar__logo" aria-hidden="true">RV</span>
          <span class="rv-sidebar__title">
            <strong>RailVision</strong>
            <span>Visão Ferroviária</span>
          </span>
        </a>
      </div>

      <div class="rv-sidebar__section-label" aria-hidden="true">Plataforma</div>

      <nav class="rv-sidebar__nav">
        @for (item of items; track item.route) {
          <a
            [routerLink]="item.route"
            routerLinkActive="rv-nav-item--active"
            class="rv-nav-item"
            [matTooltip]="isDesktopCollapsed() ? item.label : ''"
            matTooltipPosition="right"
            [attr.aria-label]="item.label"
            (click)="layout.closeMobileMenu()"
          >
            <mat-icon aria-hidden="true">{{ item.icon }}</mat-icon>
            <span class="rv-nav-item__label">{{ item.label }}</span>
          </a>
        }
      </nav>

      <div class="rv-sidebar__footer">
        <button
          type="button"
          class="rv-sidebar-toggle-btn"
          data-testid="sidebar-footer-toggle"
          (click)="layout.toggleSidebar()"
          [attr.aria-label]="layout.collapsed() ? 'Expandir barra lateral' : 'Recolher barra lateral'"
          [attr.aria-expanded]="!layout.collapsed()"
          [matTooltip]="layout.collapsed() ? 'Expandir menu lateral' : 'Recolher menu lateral'"
          matTooltipPosition="right"
        >
          <mat-icon aria-hidden="true">{{ layout.collapsed() ? 'chevron_right' : 'chevron_left' }}</mat-icon>
          <span class="rv-sidebar-toggle-btn__label">Recolher menu</span>
        </button>
      </div>
    </aside>
  `,
})
export class SidebarComponent {
  protected readonly layout = inject(LayoutStateService);
  protected readonly items = NAVIGATION_ITEMS;

  protected isDesktopCollapsed(): boolean {
    return this.layout.collapsed() && !this.layout.isMobile();
  }
}
