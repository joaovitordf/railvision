import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { RouterLink } from '@angular/router';
import { catchError, forkJoin, of } from 'rxjs';
import { InspectionSummary } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatDurationPipe } from '../../../../shared/pipes/format-duration.pipe';
import { AnnotationsApiService } from '../../../annotations/services/annotations-api.service';
import { LabApiService } from '../../../lab/services/lab-api.service';

interface ActivityEntry {
  id: string;
  icon: string;
  tone: 'info' | 'success' | 'warning' | 'error';
  title: string;
  detail: string;
  timestamp: string;
  route: string;
}

@Component({
  selector: 'rv-overview-page',
  standalone: true,
  imports: [
    RouterLink,
    DatePipe,
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    EmptyStateComponent,
    FormatDurationPipe,
  ],
  template: `
    <div class="rv-page">
      <div class="rv-page-header">
        <div>
          <h2 class="rv-page-header__title">Visão geral</h2>
          <p class="rv-page-header__subtitle">
            Indicadores reais das inspeções ferroviárias, atividades recentes e acesso rápido aos fluxos operacionais.
          </p>
        </div>

        <div class="rv-header-actions">
          <button type="button" class="rv-btn rv-btn--secondary" (click)="load()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            <span>Atualizar</span>
          </button>
          <a class="rv-btn rv-btn--primary" routerLink="/inspections" [queryParams]="{ action: 'new' }">
            <mat-icon>add</mat-icon>
            <span>Nova inspeção</span>
          </a>
        </div>
      </div>

      @if (loading()) {
        <mat-progress-bar mode="indeterminate" />
      }

      @if (errorMessage()) {
        <div class="rv-banner rv-banner--error" role="alert">
          <mat-icon aria-hidden="true">error_outline</mat-icon>
          <div>{{ errorMessage() }}</div>
        </div>
      }

      <section class="rv-kpi-grid" aria-label="Indicadores operacionais">
        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Total de inspeções</span>
            <span class="rv-kpi-card__value">{{ totalCount() }}</span>
            <span class="rv-kpi-card__meta">Vídeos registrados localmente</span>
          </div>
          <div class="rv-kpi-card__icon" style="background: var(--rv-info-bg); color: var(--rv-info-text);">
            <mat-icon>train</mat-icon>
          </div>
        </div>

        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Concluídas</span>
            <span class="rv-kpi-card__value">{{ completedCount() }}</span>
            <span class="rv-kpi-card__meta">Processamento consolidado</span>
          </div>
          <div class="rv-kpi-card__icon" style="background: var(--rv-success-bg); color: var(--rv-success-text);">
            <mat-icon>check_circle</mat-icon>
          </div>
        </div>

        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Em processamento</span>
            <span class="rv-kpi-card__value">{{ activeCount() }}</span>
            <span class="rv-kpi-card__meta">Na fila ou em execução</span>
          </div>
          <div class="rv-kpi-card__icon" style="background: var(--rv-warn-bg); color: var(--rv-warn-text);">
            <mat-icon>schedule</mat-icon>
          </div>
        </div>

        <div class="rv-kpi-card">
          <div class="rv-kpi-card__body">
            <span class="rv-kpi-card__label">Com falha</span>
            <span class="rv-kpi-card__value">{{ failedCount() }}</span>
            <span class="rv-kpi-card__meta">Prontas para reprocessamento</span>
          </div>
          <div class="rv-kpi-card__icon" style="background: var(--rv-error-bg); color: var(--rv-error-text);">
            <mat-icon>error_outline</mat-icon>
          </div>
        </div>
      </section>

      <section class="rv-action-grid" aria-label="Atalhos principais">
        <a routerLink="/inspections" [queryParams]="{ action: 'new' }" class="rv-action-card">
          <div class="rv-kpi-card__icon" style="background: var(--rv-accent-soft); color: var(--rv-accent);">
            <mat-icon>cloud_upload</mat-icon>
          </div>
          <div>
            <strong style="display: block; font-size: 0.92rem;">Iniciar inspeção</strong>
            <span style="font-size: 0.8rem; color: var(--rv-text-secondary);">Importar vídeo e contar vagões</span>
          </div>
          <mat-icon class="rv-action-card__arrow" aria-hidden="true">arrow_forward</mat-icon>
        </a>

        <a routerLink="/annotations" class="rv-action-card">
          <div class="rv-kpi-card__icon" style="background: var(--rv-success-bg); color: var(--rv-success-text);">
            <mat-icon>crop_free</mat-icon>
          </div>
          <div>
            <strong style="display: block; font-size: 0.92rem;">Abrir anotações</strong>
            <span style="font-size: 0.8rem; color: var(--rv-text-secondary);">Extrair frames e revisar caixas</span>
          </div>
          <mat-icon class="rv-action-card__arrow" aria-hidden="true">arrow_forward</mat-icon>
        </a>

        <a routerLink="/lab" class="rv-action-card">
          <div class="rv-kpi-card__icon" style="background: var(--rv-info-bg); color: var(--rv-info-text);">
            <mat-icon>science</mat-icon>
          </div>
          <div>
            <strong style="display: block; font-size: 0.92rem;">Testar modelo</strong>
            <span style="font-size: 0.8rem; color: var(--rv-text-secondary);">Avaliar checkpoints e comparar runs</span>
          </div>
          <mat-icon class="rv-action-card__arrow" aria-hidden="true">arrow_forward</mat-icon>
        </a>
      </section>

      <div class="rv-overview-grid">
        <section class="rv-card rv-card--table">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Últimas inspeções</h3>
              <p class="rv-card__subtitle">Registros operacionais mais recentes no sistema.</p>
            </div>
            <a class="rv-btn rv-btn--ghost rv-btn--sm" routerLink="/inspections">
              <span>Ver todas</span>
              <mat-icon>arrow_forward</mat-icon>
            </a>
          </div>

          @if (recentInspections().length === 0 && !loading()) {
            <rv-empty-state
              icon="train"
              title="Nenhuma inspeção registrada"
              description="Crie uma inspeção enviando um vídeo ferroviário para acompanhar resultados aqui."
            />
          } @else if (recentInspections().length > 0) {
            <div class="rv-table-wrap">
              <table class="rv-table">
                <thead>
                  <tr>
                    <th>Título</th>
                    <th>Arquivo</th>
                    <th>Duração</th>
                    <th>Vagões</th>
                    <th>Status</th>
                    <th>Data</th>
                    <th style="text-align: right;">Ação</th>
                  </tr>
                </thead>
                <tbody>
                  @for (item of recentInspections(); track item.id) {
                    <tr>
                      <td><strong>{{ item.title }}</strong></td>
                      <td class="rv-mono" style="color: var(--rv-text-secondary);">{{ item.originalFilename }}</td>
                      <td class="rv-mono">{{ item.durationMs | formatDuration }}</td>
                      <td>{{ item.status === 'COMPLETED' ? (item.wagonCount ?? 0) : '—' }}</td>
                      <td><rv-status-badge [status]="item.status" /></td>
                      <td style="color: var(--rv-text-secondary);">{{ item.createdAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                      <td style="text-align: right;">
                        <a class="rv-btn rv-btn--secondary rv-btn--sm" [routerLink]="['/inspections', item.id]">Abrir</a>
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          }
        </section>

        <section class="rv-card">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Atividades recentes</h3>
              <p class="rv-card__subtitle">Eventos reais consolidados da plataforma.</p>
            </div>
          </div>

          @if (activities().length === 0 && !loading()) {
            <rv-empty-state
              icon="history"
              title="Sem atividades recentes"
              description="As ações realizadas nas áreas de Inspeções, Anotações e Laboratório aparecem aqui."
            />
          } @else {
            <div class="rv-activity-list">
              @for (act of activities(); track act.id) {
                <a [routerLink]="act.route" class="rv-activity-item">
                  <div
                    class="rv-activity-item__icon"
                    [style.background]="
                      act.tone === 'success'
                        ? 'var(--rv-success-bg)'
                        : act.tone === 'warning'
                        ? 'var(--rv-warn-bg)'
                        : act.tone === 'error'
                        ? 'var(--rv-error-bg)'
                        : 'var(--rv-info-bg)'
                    "
                    [style.color]="
                      act.tone === 'success'
                        ? 'var(--rv-success-text)'
                        : act.tone === 'warning'
                        ? 'var(--rv-warn-text)'
                        : act.tone === 'error'
                        ? 'var(--rv-error-text)'
                        : 'var(--rv-info-text)'
                    "
                  >
                    <mat-icon aria-hidden="true">{{ act.icon }}</mat-icon>
                  </div>
                  <div style="min-width: 0; flex: 1;">
                    <div style="font-weight: 600; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                      {{ act.title }}
                    </div>
                    <div style="font-size: 0.78rem; color: var(--rv-text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                      {{ act.detail }}
                    </div>
                    <div style="font-size: 0.72rem; color: var(--rv-text-muted); margin-top: 2px;">
                      {{ act.timestamp | date: 'dd/MM/yyyy HH:mm' }}
                    </div>
                  </div>
                </a>
              }
            </div>
          }
        </section>
      </div>
    </div>
  `,
})
export class OverviewPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly labApi = inject(LabApiService, { optional: true });
  private readonly annotationsApi = inject(AnnotationsApiService, { optional: true });

  protected readonly loading = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly inspections = signal<InspectionSummary[]>([]);
  protected readonly activities = signal<ActivityEntry[]>([]);

  protected readonly totalCount = computed(() => this.inspections().length);
  protected readonly completedCount = computed(
    () => this.inspections().filter((i) => i.status === 'COMPLETED').length
  );
  protected readonly activeCount = computed(
    () => this.inspections().filter((i) => i.status === 'QUEUED' || i.status === 'PROCESSING').length
  );
  protected readonly failedCount = computed(
    () => this.inspections().filter((i) => i.status === 'FAILED').length
  );
  protected readonly recentInspections = computed(() => this.inspections().slice(0, 6));

  ngOnInit(): void {
    this.load();
  }

  protected load(): void {
    this.loading.set(true);
    this.errorMessage.set(null);

    forkJoin({
      inspections: this.api.listInspections(),
      labRuns: this.labApi ? this.labApi.listRuns().pipe(catchError(() => of([]))) : of([]),
      projects: this.annotationsApi ? this.annotationsApi.listProjects().pipe(catchError(() => of([]))) : of([]),
    }).subscribe({
      next: ({ inspections, labRuns, projects }) => {
        this.inspections.set(inspections);

        const entries: ActivityEntry[] = [];
        for (const insp of inspections.slice(0, 4)) {
          entries.push({
            id: `insp-${insp.id}`,
            icon: insp.status === 'COMPLETED' ? 'check_circle' : insp.status === 'FAILED' ? 'error_outline' : 'train',
            tone: insp.status === 'COMPLETED' ? 'success' : insp.status === 'FAILED' ? 'error' : 'info',
            title: `Inspeção: ${insp.title}`,
            detail: `${insp.originalFilename} · ${insp.status}`,
            timestamp: insp.updatedAt || insp.createdAt,
            route: `/inspections/${insp.id}`,
          });
        }
        for (const run of labRuns.slice(0, 3)) {
          entries.push({
            id: `lab-${run.runId}`,
            icon: 'science',
            tone: run.status === 'COMPLETED' ? 'info' : 'warning',
            title: `Laboratório: ${run.originalFilename}`,
            detail: `${run.modelName} · ${run.totalDetections} detecções (${run.deviceType})`,
            timestamp: run.createdAt,
            route: `/lab/runs/${run.runId}`,
          });
        }
        for (const proj of projects.slice(0, 3)) {
          entries.push({
            id: `ann-${proj.id}`,
            icon: 'crop_free',
            tone: 'success',
            title: `Anotações: ${proj.name}`,
            detail: `Campanha ${proj.campaignId} · ${proj.reviewedCount}/${proj.itemCount} revisados`,
            timestamp: proj.updatedAt,
            route: `/annotations/${proj.id}`,
          });
        }
        entries.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        this.activities.set(entries.slice(0, 6));
        this.loading.set(false);
      },
      error: (err) => {
        this.errorMessage.set(
          extractApiErrorMessage(err, 'Não foi possível carregar os dados da visão geral.')
        );
        this.loading.set(false);
      },
    });
  }
}
