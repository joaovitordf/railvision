import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ActivatedRouteSnapshot, NavigationEnd, Router, RouterLink } from '@angular/router';
import { catchError, filter, of, startWith } from 'rxjs';
import { LayoutStateService } from '../../services/layout-state.service';
import { RuntimeStatusService } from '../../services/runtime-status.service';
import { ThemeService } from '../../services/theme.service';

@Component({
  selector: 'rv-topbar',
  standalone: true,
  imports: [RouterLink, MatIconModule, MatTooltipModule],
  template: `
    <header class="rv-topbar">
      <div class="rv-topbar__left">
        <button
          type="button"
          class="rv-icon-btn"
          data-testid="topbar-menu-btn"
          (click)="layout.toggleSidebar()"
          [attr.aria-label]="menuAriaLabel()"
          [attr.aria-expanded]="menuExpanded()"
          [matTooltip]="menuTooltip()"
        >
          <mat-icon>menu</mat-icon>
        </button>

        <span class="rv-topbar__divider" aria-hidden="true"></span>

        <h1 class="rv-topbar__breadcrumb">
          <span class="rv-topbar__crumb-root">RailVision</span>
          <span class="rv-topbar__crumb-sep" aria-hidden="true">/</span>
          <span class="rv-topbar__crumb-current">{{ pageTitle() }}</span>
        </h1>
      </div>

      <div class="rv-topbar__actions">
        @if (systemChecked()) {
          <a
            routerLink="/lab/models"
            class="rv-system-pill"
            [matTooltip]="systemOnline() ? 'Sistema operacional · Diagnóstico completo no Laboratório' : 'Runtime de inferência indisponível'"
          >
            <span class="rv-status-dot" [class.rv-status-dot--offline]="!systemOnline()"></span>
            <span>{{ systemOnline() ? 'Sistema online' : 'Runtime offline' }}</span>
          </a>
        }

        <button
          type="button"
          class="rv-icon-btn"
          (click)="themeService.toggle()"
          [attr.aria-label]="themeService.theme() === 'dark' ? 'Ativar tema claro' : 'Ativar tema escuro'"
          [matTooltip]="themeService.theme() === 'dark' ? 'Tema claro' : 'Tema escuro'"
        >
          <mat-icon>{{ themeService.theme() === 'dark' ? 'light_mode' : 'dark_mode' }}</mat-icon>
        </button>
      </div>
    </header>
  `,
})
export class TopbarComponent implements OnInit {
  protected readonly layout = inject(LayoutStateService);
  protected readonly themeService = inject(ThemeService);
  private readonly runtimeStatus = inject(RuntimeStatusService, { optional: true });
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly pageTitle = signal<string>('Visão geral');
  protected readonly systemChecked = signal<boolean>(false);
  protected readonly systemOnline = signal<boolean>(false);

  constructor() {
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        startWith(null),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        const deepest = this.findDeepestSnapshot(this.router.routerState.snapshot.root);
        this.pageTitle.set((deepest.data['title'] as string | undefined) ?? 'Visão geral');
      });
  }

  ngOnInit(): void {
    if (!this.runtimeStatus) {
      return;
    }
    this.runtimeStatus
      .getStatus()
      .pipe(
        catchError(() => of(null)),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((st) => {
        this.systemChecked.set(true);
        const anyAvailable = Boolean(st?.available || st?.availableRuntimes?.some((opt) => opt.available));
        this.systemOnline.set(anyAvailable);
      });
  }

  protected menuExpanded(): boolean {
    return this.layout.isMobile() ? this.layout.mobileOpen() : !this.layout.collapsed();
  }

  protected menuAriaLabel(): string {
    if (this.layout.isMobile()) {
      return this.layout.mobileOpen() ? 'Fechar menu de navegação' : 'Abrir menu de navegação';
    }
    return this.layout.collapsed() ? 'Expandir menu lateral' : 'Recolher menu lateral';
  }

  protected menuTooltip(): string {
    if (this.layout.isMobile()) {
      return this.layout.mobileOpen() ? 'Fechar menu' : 'Abrir menu';
    }
    return this.layout.collapsed() ? 'Expandir menu lateral' : 'Recolher menu lateral';
  }

  private findDeepestSnapshot(snapshot: ActivatedRouteSnapshot): ActivatedRouteSnapshot {
    let current = snapshot;
    while (current.firstChild) {
      current = current.firstChild;
    }
    return current;
  }
}
