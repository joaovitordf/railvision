import { Component, DestroyRef, inject, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ActivatedRouteSnapshot, NavigationEnd, Router } from '@angular/router';
import { filter, startWith } from 'rxjs';
import { ThemeService } from '../../services/theme.service';

@Component({
  selector: 'rv-topbar',
  standalone: true,
  imports: [MatButtonModule, MatIconModule, MatTooltipModule],
  template: `
    <header class="rv-topbar">
      <div class="rv-topbar__left">
        <button
          type="button"
          mat-icon-button
          class="rv-mobile-menu-btn"
          (click)="toggleMobileMenu.emit()"
          aria-label="Abrir menu de navegação"
        >
          <mat-icon>menu</mat-icon>
        </button>

        <h1 class="rv-topbar__heading">
          <mat-icon aria-hidden="true">{{ pageIcon() }}</mat-icon>
          <span>{{ pageTitle() }}</span>
        </h1>
      </div>

      <div class="rv-topbar__actions">
        <button
          type="button"
          mat-icon-button
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
export class TopbarComponent {
  protected readonly themeService = inject(ThemeService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  readonly toggleMobileMenu = output<void>();
  protected readonly pageTitle = signal<string>('RailVision AI');
  protected readonly pageIcon = signal<string>('train');

  constructor() {
    this.router.events
      .pipe(
        filter((event): event is NavigationEnd => event instanceof NavigationEnd),
        startWith(null),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(() => {
        const deepest = this.findDeepestSnapshot(this.router.routerState.snapshot.root);
        this.pageTitle.set((deepest.data['title'] as string | undefined) ?? 'RailVision AI');
        this.pageIcon.set((deepest.data['icon'] as string | undefined) ?? 'train');
      });
  }

  private findDeepestSnapshot(snapshot: ActivatedRouteSnapshot): ActivatedRouteSnapshot {
    let current = snapshot;
    while (current.firstChild) {
      current = current.firstChild;
    }
    return current;
  }
}
