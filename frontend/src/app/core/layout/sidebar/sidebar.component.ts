import { Component, DestroyRef, OnInit, inject, input, output, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTooltipModule } from '@angular/material/tooltip';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { catchError, of } from 'rxjs';
import { OnnxLabApiService } from '../../../features/onnx-lab/services/onnx-lab-api.service';
import { NAVIGATION_ITEMS } from '../../navigation/navigation.config';

@Component({
  selector: 'rv-sidebar',
  standalone: true,
  imports: [RouterLink, RouterLinkActive, MatIconModule, MatButtonModule, MatTooltipModule],
  template: `
    <aside class="rv-sidebar" aria-label="Navegação principal">
      <div class="rv-sidebar__header">
        <a routerLink="/overview" class="rv-sidebar__brand" (click)="navigate.emit()">
          <span class="rv-sidebar__logo" aria-hidden="true">RV</span>
          <span class="rv-sidebar__title">
            <strong>RailVision AI</strong>
            <span>Monitoramento de Carga</span>
          </span>
        </a>

        <button
          type="button"
          mat-icon-button
          class="rv-desktop-collapse-btn"
          (click)="toggleCollapse.emit()"
          [attr.aria-label]="collapsed() ? 'Expandir barra lateral' : 'Recolher barra lateral'"
          [matTooltip]="collapsed() ? 'Expandir menu' : 'Recolher menu'"
        >
          <mat-icon>{{ collapsed() ? 'chevron_right' : 'chevron_left' }}</mat-icon>
        </button>
      </div>

      <nav class="rv-sidebar__nav">
        @for (item of items; track item.route) {
          <a
            [routerLink]="item.route"
            routerLinkActive="rv-nav-item--active"
            class="rv-nav-item"
            [matTooltip]="collapsed() ? item.label : ''"
            matTooltipPosition="right"
            (click)="navigate.emit()"
          >
            <mat-icon aria-hidden="true">{{ item.icon }}</mat-icon>
            <span class="rv-nav-item__label">{{ item.label }}</span>
          </a>
        }
      </nav>

      <div class="rv-sidebar__footer">
        <span class="rv-sidebar__env-label">{{ runtimeBadge() }}</span>
        <mat-icon aria-hidden="true" style="font-size: 16px; width: 16px; height: 16px;">memory</mat-icon>
      </div>
    </aside>
  `,
})
export class SidebarComponent implements OnInit {
  private readonly labApi = inject(OnnxLabApiService, { optional: true });
  private readonly destroyRef = inject(DestroyRef);

  readonly collapsed = input<boolean>(false);
  readonly toggleCollapse = output<void>();
  readonly navigate = output<void>();

  protected readonly items = NAVIGATION_ITEMS;
  protected readonly runtimeBadge = signal<string>('Python · FastAPI');

  ngOnInit(): void {
    if (!this.labApi) {
      return;
    }
    this.labApi
      .getStatus()
      .pipe(
        catchError(() => of(null)),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((st) => {
        if (st?.available) {
          const shortProvider = st.deviceType === 'GPU' ? 'PyTorch CUDA' : st.activeExecutionProvider;
          this.runtimeBadge.set(`${st.deviceType} · ${shortProvider}`);
        }
      });
  }
}
