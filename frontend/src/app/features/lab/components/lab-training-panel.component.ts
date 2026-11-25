import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  OnInit,
  computed,
  inject,
  output,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { Subscription, catchError, of, switchMap, timer } from 'rxjs';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { NotificationService } from '../../../core/services/notification.service';
import { EmptyStateComponent } from '../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../shared/components/status-badge/status-badge.component';
import {
  TrainingDatasetOption,
  TrainingLineagePoint,
  TrainingOverview,
  TrainingRunDetail,
  TrainingRunSummary,
} from '../models/training.models';
import { TrainingApiService } from '../services/training-api.service';
import { EpochMetricMode, TrainingChartComponent } from './training-chart.component';

@Component({
  selector: 'rv-lab-training-panel',
  standalone: true,
  imports: [
    DatePipe,
    DecimalPipe,
    FormsModule,
    MatIconModule,
    MatProgressBarModule,
    EmptyStateComponent,
    StatusBadgeComponent,
    TrainingChartComponent,
  ],
  template: `
    <div class="rv-page">
      <!-- Config & Version Launch Card -->
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">
              {{ overview()?.lineageName || 'Detector de Vagões Abertos' }} — Evolução Contínua
            </h3>
            <p class="rv-card__subtitle">
              Cada nova rodada parte automaticamente da versão mais recente e grava métricas em validação sem tocar no conjunto de teste congelado.
            </p>
          </div>

          <div class="rv-header-actions">
            @if (overview(); as ov) {
              <span
                class="rv-badge"
                [class.rv-badge--success]="ov.gpuAvailable"
                [class.rv-badge--error]="!ov.gpuAvailable"
              >
                <span class="rv-badge__dot" aria-hidden="true"></span>
                {{ ov.gpuAvailable ? (ov.gpuName || 'GPU CUDA') : 'GPU indisponível' }}
              </span>
              <span class="rv-badge rv-badge--info" data-testid="next-version-badge">
                Próxima versão: {{ ov.nextVersionLabel }}
              </span>
            }
            <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="refreshAll()">
              <mat-icon>refresh</mat-icon>
              <span>Atualizar</span>
            </button>
          </div>
        </div>

        @if (errorMessage()) {
          <div class="rv-banner rv-banner--error" role="alert" style="margin-bottom: 14px;">
            <mat-icon aria-hidden="true">error_outline</mat-icon>
            <div>{{ errorMessage() }}</div>
          </div>
        }

        @if (activeRun(); as act) {
          <div class="rv-banner rv-banner--warning" role="status" style="margin-bottom: 14px;" data-testid="gpu-training-active-banner">
            <mat-icon aria-hidden="true">warning</mat-icon>
            <div>
              <strong>Treinamento ativo na RTX 4060 Ti ({{ act.versionLabel }} · {{ act.progressPercent }}%):</strong>
              GPU compartilhada (8 GB VRAM) — evite executar inferências pesadas no Laboratório ou Estúdio durante o treino.
            </div>
          </div>
        }

        @if (overview(); as ov) {
          <div class="rv-banner rv-banner--info" style="margin-bottom: 16px;">
            <mat-icon aria-hidden="true">info</mat-icon>
            <div>
              <strong>Base automática encadeada:</strong>
              <span class="rv-mono" data-testid="recommended-base-label">{{ ov.recommendedBaseLabel }}</span>.
              O conjunto <span class="rv-mono">test</span> (<span class="rv-mono">vid_09</span>/<span class="rv-mono">vid_16</span>) permanece congelado e nunca é executado por esta tela.
            </div>
          </div>

          @if (selectedDataset(); as ds) {
            @if (ds.warning) {
              <div class="rv-banner rv-banner--warning" style="margin-bottom: 16px;" data-testid="val-campaign-warning">
                <mat-icon aria-hidden="true">warning</mat-icon>
                <div>{{ ds.warning }}</div>
              </div>
            }
          }

          <form (ngSubmit)="startTraining()" style="display: flex; flex-direction: column; gap: 14px;">
            <div class="rv-form-grid">
              <div class="rv-field">
                <label class="rv-label" for="trn-dataset">Dataset de treinamento</label>
                <select
                  id="trn-dataset"
                  class="rv-select"
                  name="datasetId"
                  data-testid="training-dataset-select"
                  [ngModel]="datasetId()"
                  (ngModelChange)="datasetId.set($event)"
                >
                  @for (ds of ov.datasets; track ds.id) {
                    <option [ngValue]="ds.id">
                      {{ ds.name }} ({{ ds.trainCount }} train / {{ ds.valCount }} val)
                    </option>
                  }
                </select>
              </div>

              <div class="rv-field" style="max-width: 150px;">
                <label class="rv-label" for="trn-epochs">Épocas</label>
                <input
                  id="trn-epochs"
                  class="rv-input"
                  type="number"
                  min="1"
                  max="200"
                  name="epochs"
                  data-testid="training-epochs-input"
                  [ngModel]="epochs()"
                  (ngModelChange)="epochs.set(+$event)"
                />
              </div>

              <div class="rv-field">
                <label class="rv-label" for="trn-notes">Nota da rodada (opcional)</label>
                <input
                  id="trn-notes"
                  class="rv-input"
                  type="text"
                  maxlength="500"
                  name="notes"
                  placeholder="Ex.: +40 frames de ponte elevada + 12 negativos"
                  [ngModel]="notes()"
                  (ngModelChange)="notes.set($event)"
                />
              </div>
            </div>

            <details class="rv-advanced-box">
              <summary class="rv-advanced-summary">
                <mat-icon>tune</mat-icon>
                <span>Opções avançadas (substituir modelo base, resolução e hiperparâmetros)</span>
              </summary>
              <div class="rv-form-grid" style="margin-top: 12px;">
                <div class="rv-field">
                  <label class="rv-label" for="trn-base-override">Ponto de partida (opcional)</label>
                  <select
                    id="trn-base-override"
                    class="rv-select"
                    name="baseOverride"
                    data-testid="training-base-override"
                    [ngModel]="baseOverride()"
                    (ngModelChange)="baseOverride.set($event)"
                  >
                    @for (opt of ov.advancedBaseOptions; track opt.id) {
                      <option [ngValue]="opt.id">{{ opt.label }}</option>
                    }
                  </select>
                </div>

                <div class="rv-field">
                  <label class="rv-label" for="trn-imgsz">Resolução (imgsz)</label>
                  <select id="trn-imgsz" class="rv-select" name="imgsz" [ngModel]="imgsz()" (ngModelChange)="imgsz.set(+$event)">
                    <option [ngValue]="640">640 px (Padrão RTX 4060 Ti)</option>
                    <option [ngValue]="768">768 px (Mais detalhe)</option>
                    <option [ngValue]="960">960 px (Alta resolução)</option>
                    <option [ngValue]="512">512 px (Rápido)</option>
                  </select>
                </div>

                <div class="rv-field">
                  <label class="rv-label" for="trn-batch">Batch size</label>
                  <input id="trn-batch" class="rv-input" type="number" min="1" max="64" name="batch" [ngModel]="batch()" (ngModelChange)="batch.set(+$event)" />
                </div>

                <div class="rv-field">
                  <label class="rv-label" for="trn-freeze">Camadas congeladas (freeze)</label>
                  <input id="trn-freeze" class="rv-input" type="number" min="0" max="30" name="freeze" [ngModel]="freeze()" (ngModelChange)="freeze.set(+$event)" />
                </div>

                <div class="rv-field">
                  <label class="rv-label" for="trn-patience">Paciência (early stop)</label>
                  <input id="trn-patience" class="rv-input" type="number" min="3" max="100" name="patience" [ngModel]="patience()" (ngModelChange)="patience.set(+$event)" />
                </div>

                <div class="rv-field">
                  <label class="rv-label" for="trn-lr0">Learning rate inicial (lr0)</label>
                  <input id="trn-lr0" class="rv-input" type="number" min="0.00001" max="0.05" step="0.0001" name="lr0" [ngModel]="lr0()" (ngModelChange)="lr0.set(+$event)" />
                </div>
              </div>
            </details>

            <div class="rv-header-actions">
              @if (activeRun(); as running) {
                <button
                  type="button"
                  class="rv-btn rv-btn--danger"
                  data-testid="stop-training-btn"
                  [disabled]="stopping()"
                  (click)="stopActiveRun(running.runId)"
                >
                  <mat-icon>stop</mat-icon>
                  <span>{{ stopping() ? 'Parando...' : 'Parar treino (' + running.versionLabel + ')' }}</span>
                </button>
              } @else {
                <button
                  type="submit"
                  class="rv-btn rv-btn--primary"
                  data-testid="start-training-btn"
                  [disabled]="!canStart()"
                >
                  <mat-icon>play_arrow</mat-icon>
                  <span>{{ starting() ? 'Iniciando...' : 'Treinar versão ' + ov.nextVersionLabel }}</span>
                </button>
              }
            </div>
          </form>
        }
      </section>

      <!-- Selected/Active Run Live Progress Strip -->
      @if (selectedRunDetail(); as run) {
        <section class="rv-kpi-grid" aria-label="Indicadores da rodada selecionada">
          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">Rodada selecionada</span>
              <span class="rv-kpi-card__value" style="font-size: 1.55rem;">{{ run.versionLabel }}</span>
              <span class="rv-kpi-card__meta rv-mono">Base: {{ run.baseLabel }}</span>
            </div>
            <rv-status-badge [status]="run.status" />
          </div>

          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">Épocas</span>
              <span class="rv-kpi-card__value" style="font-size: 1.55rem;">
                {{ run.currentEpoch }} / {{ run.epochsRequested }}
              </span>
              <span class="rv-kpi-card__meta">{{ run.progressPercent }}% · imgsz {{ run.imgsz }} · batch {{ run.batch }}</span>
            </div>
          </div>

          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">Melhor Val mAP50</span>
              <span class="rv-kpi-card__value" style="font-size: 1.55rem;" data-testid="training-kpi-map50">
                {{ bestOrFinalMap50(run) != null ? ((bestOrFinalMap50(run)! * 100) | number: '1.1-1') + '%' : '—' }}
              </span>
              <span class="rv-kpi-card__meta">
                mAP50-95: {{ bestOrFinalMap95(run) != null ? ((bestOrFinalMap95(run)! * 100) | number: '1.1-1') + '%' : '—' }}
              </span>
            </div>
          </div>

          <div class="rv-kpi-card">
            <div class="rv-kpi-card__body">
              <span class="rv-kpi-card__label">GPU / Duração</span>
              <span class="rv-kpi-card__value" style="font-size: 1.45rem;">
                {{ run.peakVramMib != null ? (run.peakVramMib | number: '1.0-0') + ' MiB' : 'RTX 4060 Ti' }}
              </span>
              <span class="rv-kpi-card__meta">
                {{ run.wallTimeS != null ? (run.wallTimeS | number: '1.1-1') + ' s' : 'Em acompanhamento' }}
              </span>
            </div>
          </div>
        </section>

        @if (run.status === 'QUEUED' || run.status === 'RUNNING') {
          <section class="rv-card" style="padding: 14px 20px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.84rem; margin-bottom: 8px;">
              <span>Executando <span class="rv-mono">{{ run.experimentName }}</span> (polling automático a cada 2s)</span>
              <strong>{{ run.progressPercent }}%</strong>
            </div>
            <mat-progress-bar
              [mode]="run.progressPercent > 0 ? 'determinate' : 'indeterminate'"
              [value]="run.progressPercent"
            />
          </section>
        }
      }

      <!-- Two-Chart Comparison Grid -->
      <section class="rv-two-col-grid">
        <div class="rv-card">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Evolução entre versões (v0 → {{ overview()?.nextVersionLabel || 'v1' }})</h3>
              <p class="rv-card__subtitle">
                Comparação de progresso acumulado entre rodadas; linha tracejada indica mudança no conjunto de validação.
              </p>
            </div>
          </div>

          <rv-training-chart
            mode="lineage"
            [lineagePoints]="overview()?.lineagePoints ?? []"
            [baselineMap50]="overview()?.baselineMetrics?.map50 ?? null"
            ariaLabel="Evolução das versões do modelo"
            (lineageSelected)="onLineagePointClick($event)"
          />
        </div>

        <div class="rv-card">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">
                Curvas por época {{ selectedRunDetail() ? '— ' + selectedRunDetail()!.versionLabel : '' }}
              </h3>
              <p class="rv-card__subtitle">Leitura contínua do arquivo <span class="rv-mono">results.csv</span>.</p>
            </div>

            <div class="rv-header-actions">
              <div class="rv-seg-group" role="group" aria-label="Tipo de curva por época">
                <button
                  type="button"
                  class="rv-btn rv-btn--sm"
                  [class.rv-btn--primary]="epochChartView() === 'metrics'"
                  [class.rv-btn--secondary]="epochChartView() !== 'metrics'"
                  (click)="epochChartView.set('metrics')"
                >
                  Métricas
                </button>
                <button
                  type="button"
                  class="rv-btn rv-btn--sm"
                  [class.rv-btn--primary]="epochChartView() === 'losses'"
                  [class.rv-btn--secondary]="epochChartView() !== 'losses'"
                  (click)="epochChartView.set('losses')"
                >
                  Losses
                </button>
              </div>

              @if (otherCompletedRuns().length > 0 && epochChartView() === 'metrics') {
                <select
                  class="rv-select"
                  style="width: 150px; height: 32px; font-size: 0.78rem;"
                  aria-label="Sobrepor rodada anterior"
                  [ngModel]="compareRunId()"
                  (ngModelChange)="onCompareRunChange($event)"
                >
                  <option [ngValue]="null">Sem overlay</option>
                  @for (cr of otherCompletedRuns(); track cr.runId) {
                    <option [ngValue]="cr.runId">Comparar {{ cr.versionLabel }}</option>
                  }
                </select>
              }
            </div>
          </div>

          <rv-training-chart
            mode="epochs"
            [epochMetricMode]="epochChartView()"
            [epochHistory]="selectedRunDetail()?.epochHistory ?? []"
            [compareEpochHistory]="compareRunDetail()?.epochHistory ?? []"
            [compareLabel]="compareRunDetail()?.versionLabel ?? null"
            [baselineMap50]="overview()?.baselineMetrics?.map50 ?? null"
            ariaLabel="Métricas e perdas por época"
          />
        </div>
      </section>

      <!-- Log Tail (Collapsible) -->
      @if (selectedRunDetail(); as run) {
        @if (run.logTail.length > 0) {
          <section class="rv-card" style="padding: 14px 20px;">
            <details [open]="run.status === 'FAILED' || run.status === 'RUNNING'">
              <summary class="rv-advanced-summary">
                <mat-icon>terminal</mat-icon>
                <span>Log do processo de treinamento ({{ run.versionLabel }} · últimas {{ run.logTail.length }} linhas)</span>
              </summary>
              <pre class="rv-log-pre rv-mono">{{ run.logTail.join('\n') }}</pre>
            </details>
          </section>
        }
      }

      <!-- Versions History Table -->
      <section class="rv-card rv-card--table">
        <div class="rv-card__header">
          <div>
            <h3 class="rv-card__title">Histórico da linha de versões ({{ overview()?.runs?.length ?? 0 }})</h3>
            <p class="rv-card__subtitle">
              Clique em uma versão para ver curvas detalhadas, registrar no Model Registry ou testar na aba Inferência.
            </p>
          </div>
        </div>

        @if (!overview()?.runs?.length) {
          <rv-empty-state
            icon="trending_up"
            title="Nenhuma rodada treinada pela interface ainda"
            description="O baseline v0 (exp04c) já aparece no gráfico acima. Clique em 'Treinar versão v1' para iniciar a evolução do seu modelo."
          />
        } @else {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Versão</th>
                  <th>Base usada</th>
                  <th>Dataset</th>
                  <th>Épocas</th>
                  <th>mAP50</th>
                  <th>mAP50-95</th>
                  <th>Status</th>
                  <th>Data</th>
                  <th>Ações</th>
                </tr>
              </thead>
              <tbody>
                @for (item of overview()!.runs; track item.runId) {
                  <tr
                    class="rv-row--selectable"
                    [class.rv-row--selected]="selectedRunId() === item.runId"
                    (click)="selectRun(item.runId)"
                  >
                    <td>
                      <strong class="rv-mono">{{ item.versionLabel }}</strong>
                      @if (item.notes) {
                        <div style="font-size: 0.76rem; color: var(--rv-text-muted); max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                          {{ item.notes }}
                        </div>
                      }
                    </td>
                    <td class="rv-mono">{{ item.baseLabel }}</td>
                    <td class="rv-mono">{{ item.datasetId }}</td>
                    <td>{{ item.epochsCompleted }} / {{ item.epochsRequested }}</td>
                    <td class="rv-mono">
                      {{ bestOrFinalMap50(item) != null ? ((bestOrFinalMap50(item)! * 100) | number: '1.1-1') + '%' : '—' }}
                    </td>
                    <td class="rv-mono">
                      {{ bestOrFinalMap95(item) != null ? ((bestOrFinalMap95(item)! * 100) | number: '1.1-1') + '%' : '—' }}
                    </td>
                    <td><rv-status-badge [status]="item.status" /></td>
                    <td>{{ item.createdAt | date: 'dd/MM HH:mm' }}</td>
                    <td (click)="$event.stopPropagation()">
                      <div class="rv-header-actions">
                        @if (item.bestWeightsPath && !item.registeredModelId && item.status !== 'RUNNING' && item.status !== 'QUEUED') {
                          <button
                            type="button"
                            class="rv-btn rv-btn--secondary rv-btn--sm"
                            data-testid="register-training-checkpoint-btn"
                            [disabled]="registeringId() === item.runId"
                            (click)="registerRun(item.runId)"
                          >
                            <mat-icon>add_circle_outline</mat-icon>
                            <span>{{ registeringId() === item.runId ? 'Registrando...' : 'Registrar candidato' }}</span>
                          </button>
                        } @else if (item.registeredModelId) {
                          <button
                            type="button"
                            class="rv-btn rv-btn--ghost rv-btn--sm"
                            (click)="testRegisteredModel.emit(item.registeredModelId!)"
                          >
                            <mat-icon>play_arrow</mat-icon>
                            <span>Testar modelo</span>
                          </button>
                        }
                      </div>
                    </td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>
    </div>
  `,
  styles: [
    `
      .rv-advanced-box {
        border: 1px solid var(--rv-border);
        border-radius: var(--rv-radius-sm);
        background: var(--rv-bg-surface-muted);
        padding: 10px 14px;
      }

      .rv-advanced-summary {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        cursor: pointer;
        font-size: 0.83rem;
        font-weight: 600;
        color: var(--rv-text-secondary);
        user-select: none;
      }

      .rv-seg-group {
        display: inline-flex;
        gap: 4px;
      }

      .rv-log-pre {
        margin: 10px 0 0;
        padding: 12px;
        border-radius: var(--rv-radius-sm);
        background: #060b13;
        color: #cbd5e1;
        font-size: 0.75rem;
        line-height: 1.45;
        max-height: 220px;
        overflow: auto;
        white-space: pre-wrap;
      }
    `,
  ],
})
export class LabTrainingPanelComponent implements OnInit {
  private readonly api = inject(TrainingApiService);
  private readonly notifications = inject(NotificationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly checkpointRegistered = output<string>();
  readonly testRegisteredModel = output<string>();

  protected readonly overview = signal<TrainingOverview | null>(null);
  protected readonly selectedRunId = signal<string | null>(null);
  protected readonly selectedRunDetail = signal<TrainingRunDetail | null>(null);
  protected readonly compareRunId = signal<string | null>(null);
  protected readonly compareRunDetail = signal<TrainingRunDetail | null>(null);
  protected readonly epochChartView = signal<EpochMetricMode>('metrics');

  protected readonly datasetId = signal<string>('open-top-yolo-v1');
  protected readonly baseOverride = signal<string>('auto');
  protected readonly epochs = signal<number>(30);
  protected readonly patience = signal<number>(15);
  protected readonly imgsz = signal<number>(640);
  protected readonly batch = signal<number>(8);
  protected readonly freeze = signal<number>(10);
  protected readonly lr0 = signal<number>(0.0005);
  protected readonly notes = signal<string>('');

  protected readonly starting = signal(false);
  protected readonly stopping = signal(false);
  protected readonly registeringId = signal<string | null>(null);
  protected readonly errorMessage = signal<string | null>(null);

  private pollSub: Subscription | null = null;

  protected readonly activeRun = computed<TrainingRunSummary | null>(() => {
    const ov = this.overview();
    if (!ov) return null;
    return ov.runs.find((r) => r.status === 'QUEUED' || r.status === 'RUNNING') ?? null;
  });

  protected readonly selectedDataset = computed<TrainingDatasetOption | null>(() => {
    const ov = this.overview();
    if (!ov) return null;
    return ov.datasets.find((d) => d.id === this.datasetId()) ?? ov.datasets[0] ?? null;
  });

  protected readonly otherCompletedRuns = computed<TrainingRunSummary[]>(() => {
    const ov = this.overview();
    const curId = this.selectedRunId();
    if (!ov) return [];
    return ov.runs.filter((r) => r.status === 'COMPLETED' && r.runId !== curId);
  });

  protected readonly canStart = computed(() => {
    const ov = this.overview();
    return Boolean(ov?.gpuAvailable && this.selectedDataset() && !this.activeRun() && !this.starting());
  });

  constructor() {
    this.destroyRef.onDestroy(() => this.stopPolling());
  }

  ngOnInit(): void {
    this.refreshAll();
  }

  protected refreshAll(): void {
    this.errorMessage.set(null);
    this.api
      .getOverview()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (ov) => {
          this.overview.set(ov);
          if (ov.datasets.length > 0 && !ov.datasets.some((d) => d.id === this.datasetId())) {
            this.datasetId.set(ov.datasets[0].id);
          }
          const targetId = this.selectedRunId() || ov.activeRunId || ov.runs[0]?.runId || null;
          if (targetId) {
            this.selectRun(targetId);
          }
          if (ov.activeRunId) {
            this.startPolling(ov.activeRunId);
          }
        },
        error: (err) => {
          this.errorMessage.set(extractApiErrorMessage(err, 'Falha ao carregar painel de treinamento.'));
        },
      });
  }

  protected selectRun(runId: string): void {
    this.selectedRunId.set(runId);
    this.api
      .getRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (det) => {
          this.selectedRunDetail.set(det);
          if (det.status === 'QUEUED' || det.status === 'RUNNING') {
            this.startPolling(det.runId);
          }
        },
        error: (err) => {
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao carregar detalhes da rodada.'));
        },
      });
  }

  protected onLineagePointClick(pt: TrainingLineagePoint): void {
    if (pt.runId) {
      this.selectRun(pt.runId);
    }
  }

  protected onCompareRunChange(runId: string | null): void {
    this.compareRunId.set(runId);
    if (!runId) {
      this.compareRunDetail.set(null);
      return;
    }
    this.api
      .getRun(runId)
      .pipe(catchError(() => of(null)), takeUntilDestroyed(this.destroyRef))
      .subscribe((det) => this.compareRunDetail.set(det));
  }

  protected startTraining(): void {
    if (!this.canStart()) return;
    this.starting.set(true);
    this.errorMessage.set(null);
    const override = this.baseOverride();

    this.api
      .startRun({
        datasetId: this.datasetId(),
        baseOverride: override === 'auto' ? null : override,
        epochs: Math.max(1, Math.min(200, Number(this.epochs()) || 30)),
        patience: Math.max(3, Math.min(100, Number(this.patience()) || 15)),
        imgsz: Number(this.imgsz()) || 640,
        batch: Math.max(1, Math.min(64, Number(this.batch()) || 8)),
        freeze: Math.max(0, Math.min(30, Number(this.freeze()) || 0)),
        lr0: Number(this.lr0()) || 0.0005,
        cosLr: true,
        notes: this.notes().trim() || null,
      })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => {
          this.starting.set(false);
          this.notes.set('');
          this.selectedRunId.set(run.runId);
          this.notifications.success(`Treinamento ${run.versionLabel} iniciado na GPU.`);
          this.refreshAll();
          this.startPolling(run.runId);
        },
        error: (err) => {
          this.starting.set(false);
          const msg = extractApiErrorMessage(err, 'Falha ao iniciar treinamento.');
          this.errorMessage.set(msg);
          this.notifications.error(msg);
        },
      });
  }

  protected stopActiveRun(runId: string): void {
    this.stopping.set(true);
    this.api
      .stopRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (stopped) => {
          this.stopping.set(false);
          this.stopPolling();
          this.notifications.info(`Treinamento ${stopped.versionLabel} interrompido.`);
          this.refreshAll();
        },
        error: (err) => {
          this.stopping.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao parar treinamento.'));
        },
      });
  }

  protected registerRun(runId: string): void {
    this.registeringId.set(runId);
    this.api
      .registerCheckpoint(runId, { activate: false })
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (cp) => {
          this.registeringId.set(null);
          this.notifications.success(`Checkpoint '${cp.id}' registrado como candidato no Model Registry.`);
          this.checkpointRegistered.emit(cp.id);
          this.refreshAll();
        },
        error: (err) => {
          this.registeringId.set(null);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao registrar checkpoint.'));
        },
      });
  }

  protected bestOrFinalMap50(run: TrainingRunSummary): number | null {
    return run.finalMap50 ?? run.bestMap50;
  }

  protected bestOrFinalMap95(run: TrainingRunSummary): number | null {
    return run.finalMap5095 ?? run.bestMap5095;
  }

  private startPolling(runId: string): void {
    this.stopPolling();
    this.pollSub = timer(1500, 2000)
      .pipe(
        switchMap(() => this.api.getRun(runId).pipe(catchError(() => of(null)))),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe((det) => {
        if (!det) {
          return;
        }
        if (this.selectedRunId() === runId) {
          this.selectedRunDetail.set(det);
        }
        this.overview.update((ov) =>
          ov
            ? {
                ...ov,
                runs: ov.runs.map((r) => (r.runId === det.runId ? { ...r, ...det } : r)),
              }
            : ov
        );
        if (det.status !== 'QUEUED' && det.status !== 'RUNNING') {
          this.stopPolling();
          this.refreshAll();
          if (det.status === 'COMPLETED') {
            this.notifications.success(`Treinamento ${det.versionLabel} concluído!`);
          } else if (det.status === 'FAILED') {
            this.notifications.error(det.errorMessage || `Treinamento ${det.versionLabel} falhou.`);
          }
        }
      });
  }

  private stopPolling(): void {
    this.pollSub?.unsubscribe();
    this.pollSub = null;
  }
}
