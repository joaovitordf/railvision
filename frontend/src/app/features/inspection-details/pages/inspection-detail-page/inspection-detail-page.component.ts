import { DatePipe, DecimalPipe } from '@angular/common';
import {
  Component,
  DestroyRef,
  ElementRef,
  OnInit,
  ViewChild,
  computed,
  inject,
  input,
  signal,
} from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { Subscription, switchMap, takeWhile, timer } from 'rxjs';
import {
  InspectionDetail,
  ProcessingJob,
  WagonObservation,
  WagonResult,
} from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { DemoFixtureBannerComponent } from '../../../../shared/components/demo-fixture-banner/demo-fixture-banner.component';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';
import { LoadingLevelLabelPipe } from '../../../../shared/pipes/loading-level-label.pipe';
import { MaterialLabelPipe } from '../../../../shared/pipes/material-label.pipe';

@Component({
  selector: 'rv-inspection-detail-page',
  standalone: true,
  imports: [
    RouterLink,
    DatePipe,
    DecimalPipe,
    MatButtonModule,
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    DemoFixtureBannerComponent,
    EmptyStateComponent,
    FormatBytesPipe,
    MaterialLabelPipe,
    LoadingLevelLabelPipe,
  ],
  template: `
    <div class="rv-page">
      <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <a mat-stroked-button routerLink="/inspections">
          <mat-icon>arrow_back</mat-icon>
          Voltar para inspeções
        </a>

        @if (inspection(); as detail) {
          <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
            <rv-status-badge [status]="detail.status" />
            <button
              type="button"
              mat-flat-button
              color="primary"
              [disabled]="isProcessing() || startingProcess()"
              (click)="startOrReprocess()"
            >
              <mat-icon>play_arrow</mat-icon>
              {{ detail.status === 'UPLOADED' ? 'Iniciar processamento' : 'Reprocessar' }}
            </button>
          </div>
        }
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

      @if (inspection(); as detail) {
        @if (showDemoBanner()) {
          <rv-demo-fixture-banner [engineType]="activeEngineType()" />
        }

        @if (activeJob(); as job) {
          @if (job.status === 'QUEUED' || job.status === 'RUNNING') {
            <section class="rv-card" data-testid="job-progress-card">
              <div class="rv-card__header">
                <div>
                  <h2 class="rv-card__title">Processamento em andamento</h2>
                  <p class="rv-card__subtitle">
                    Motor <span class="rv-mono">{{ job.engineType }}</span> · Job
                    <span class="rv-mono">{{ job.id }}</span>
                  </p>
                </div>
                <rv-status-badge [status]="job.status" />
              </div>
              <mat-progress-bar
                [mode]="job.progressPercent != null ? 'determinate' : 'indeterminate'"
                [value]="job.progressPercent ?? 0"
              />
            </section>
          }

          @if (job.status === 'FAILED') {
            <div class="rv-banner rv-banner--error" role="alert" data-testid="job-failed-banner">
              <mat-icon aria-hidden="true">report_problem</mat-icon>
              <div style="display: flex; flex-direction: column; gap: 6px;">
                <strong>
                  Falha no processamento
                  @if (job.errorCode) {
                    (<span class="rv-mono">{{ job.errorCode }}</span>)
                  }
                </strong>
                <span>{{ job.errorMessage ?? 'Ocorreu uma falha durante a execução do job.' }}</span>
              </div>
            </div>
          }
        }

        <section
          style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px;"
        >
          <div class="rv-card">
            <div class="rv-card__header">
              <div>
                <h2 class="rv-card__title">{{ detail.title }}</h2>
                <p class="rv-card__subtitle">
                  Arquivo <span class="rv-mono">{{ detail.originalFilename }}</span> ({{ detail.fileSizeBytes | formatBytes }})
                </p>
              </div>
            </div>

            <div class="rv-video-stage">
              <video
                #videoEl
                controls
                preload="metadata"
                [src]="mediaUrl()"
                (loadedmetadata)="redrawOverlays()"
                (timeupdate)="redrawOverlays()"
              ></video>
              <canvas #overlayCanvas></canvas>
            </div>

            <div style="margin-top: 12px; font-size: 0.82rem; color: var(--rv-text-muted); display: flex; flex-wrap: wrap; gap: 16px;">
              <span>MIME: <strong class="rv-mono">{{ detail.contentType }}</strong></span>
              @if (detail.widthPx && detail.heightPx) {
                <span>
                  Vídeo: <strong class="rv-mono">{{ detail.widthPx }}×{{ detail.heightPx }}</strong>
                  @if (detail.fps) {
                    · <strong class="rv-mono">{{ detail.fps | number: '1.0-2' }} fps</strong>
                  }
                  @if (detail.codec) {
                    · <strong class="rv-mono">{{ detail.codec }}</strong>
                  }
                </span>
              }
              <span>SHA-256: <strong class="rv-mono">{{ detail.sha256Checksum.slice(0, 16) }}…</strong></span>
              <span>Criada em: <strong>{{ detail.createdAt | date: 'dd/MM/yyyy HH:mm:ss' }}</strong></span>
            </div>
          </div>

          <div class="rv-card">
            <div class="rv-card__header">
              <div>
                <h2 class="rv-card__title">Resumo da inspeção</h2>
                <p class="rv-card__subtitle">Consolidação por vagão da última execução concluída.</p>
              </div>
            </div>

            <div class="rv-kpi-grid" style="grid-template-columns: repeat(2, minmax(120px, 1fr));">
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Vagões registrados</span>
                <span class="rv-kpi-card__value">{{ wagons().length }}</span>
              </div>
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Sinalizados para revisão</span>
                <span class="rv-kpi-card__value">{{ reviewCount() }}</span>
              </div>
            </div>

            @if (detail.notes) {
              <p style="margin-top: 16px; font-size: 0.9rem; color: var(--rv-text-secondary);">
                <strong>Observações:</strong> {{ detail.notes }}
              </p>
            }

            <div style="margin-top: 16px; font-size: 0.84rem; color: var(--rv-text-secondary);">
              <strong>Escopo dos resultados:</strong> Detecção (<span class="rv-mono">open_top_freight_wagon</span>) e
              rastreamento (<span class="rv-mono">ByteTrack</span>) reais no vídeo original; percentual numérico de carga permanece
              <span class="rv-mono">null</span> (<span class="rv-mono">physicalVolumeVerified = false</span>).
            </div>
          </div>
        </section>

        <section class="rv-card">
          <div class="rv-card__header">
            <div>
              <h2 class="rv-card__title">Vagões consolidados</h2>
              <p class="rv-card__subtitle">
                Clique em uma linha para inspecionar observações, overlays e distribuição de votos.
              </p>
            </div>
          </div>

          @if (wagons().length === 0) {
            <rv-empty-state
              icon="view_carousel"
              [title]="detail.status === 'COMPLETED' ? 'Nenhum vagão aberto detectado' : 'Sem resultados de vagões disponíveis'"
              [description]="
                detail.status === 'COMPLETED'
                  ? 'Nenhum vagão graneleiro aberto foi detectado neste vídeo pelo detector atual.'
                  : 'Inicie o processamento para detectar e consolidar os vagões presentes no vídeo.'
              "
            />
          } @else {
            <div class="rv-table-wrap">
              <table class="rv-table">
                <thead>
                  <tr>
                    <th>Seq.</th>
                    <th>Track ID (temporário)</th>
                    <th>Material</th>
                    <th>Nível visual</th>
                    <th>Observações</th>
                    <th>Revisão</th>
                    <th>Origem</th>
                  </tr>
                </thead>
                <tbody>
                  @for (wagon of wagons(); track wagon.id) {
                    <tr
                      class="rv-row--selectable"
                      [class.rv-row--selected]="selectedWagon()?.id === wagon.id"
                      (click)="selectWagon(wagon)"
                    >
                      <td><strong>#{{ wagon.wagonIdentity.sequenceIndex }}</strong></td>
                      <td class="rv-mono">{{ wagon.wagonIdentity.trackId }}</td>
                      <td>
                        {{ (wagon.material ? wagon.material.materialClass : null) | materialLabel }}
                        @if (wagon.material?.confidence != null) {
                          <span style="font-size: 0.78rem; color: var(--rv-text-muted);">
                            ({{ wagon.material!.confidence! | number: '1.2-2' }})
                          </span>
                        }
                      </td>
                      <td>
                        {{ wagon.loadingEstimate.level | loadingLevelLabel }}
                        @if (wagon.loadingEstimate.confidence != null) {
                          <span style="font-size: 0.78rem; color: var(--rv-text-muted);">
                            ({{ wagon.loadingEstimate.confidence | number: '1.2-2' }})
                          </span>
                        }
                      </td>
                      <td>{{ wagon.consolidation.observationsCount }}</td>
                      <td>
                        @if (wagon.reviewRequired) {
                          <span class="rv-badge rv-badge--warning" data-testid="review-required-badge">
                            Revisão necessária
                          </span>
                        } @else {
                          <span class="rv-badge rv-badge--success">OK</span>
                        }
                      </td>
                      <td>
                        @if (wagon.isExample) {
                          <span class="rv-badge rv-badge--warning">Exemplo fictício</span>
                        } @else {
                          <span class="rv-badge rv-badge--info">Inferência</span>
                        }
                      </td>
                    </tr>
                  }
                </tbody>
              </table>
            </div>
          }
        </section>

        @if (selectedWagon(); as wagon) {
          <section class="rv-card">
            <div class="rv-card__header">
              <div>
                <h2 class="rv-card__title">
                  Detalhes do vagão #{{ wagon.wagonIdentity.sequenceIndex }}
                  (<span class="rv-mono">{{ wagon.wagonIdentity.trackId }}</span>)
                </h2>
                <p class="rv-card__subtitle">
                  Identificação física (OCR): <span class="rv-mono">{{ wagon.wagonIdentity.physicalId ?? 'null (fora do MVP)' }}</span>
                </p>
              </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 16px;">
              <div>
                <strong>Motivos de revisão</strong>
                @if (wagon.reviewReasons.length === 0) {
                  <div style="color: var(--rv-text-muted); font-size: 0.88rem;">Nenhum sinalizado.</div>
                } @else {
                  <ul style="margin: 6px 0 0; padding-left: 18px; font-size: 0.88rem;">
                    @for (reason of wagon.reviewReasons; track reason) {
                      <li class="rv-mono">{{ reason }}</li>
                    }
                  </ul>
                }
              </div>

              <div>
                <strong>Discordâncias registradas</strong>
                @if (wagon.consolidation.disagreements.length === 0) {
                  <div style="color: var(--rv-text-muted); font-size: 0.88rem;">Nenhuma discordância.</div>
                } @else {
                  <ul style="margin: 6px 0 0; padding-left: 18px; font-size: 0.88rem;">
                    @for (disagreement of wagon.consolidation.disagreements; track disagreement) {
                      <li>{{ disagreement }}</li>
                    }
                  </ul>
                }
              </div>

              <div>
                <strong>Versões de modelo associadas</strong>
                <div style="margin-top: 6px; font-size: 0.82rem;">
                  @for (entry of modelVersionEntries(wagon); track entry[0]) {
                    <div>
                      <span class="rv-mono">{{ entry[0] }}</span>: <span class="rv-mono">{{ entry[1] }}</span>
                    </div>
                  }
                </div>
              </div>
            </div>

            <h3 style="margin: 0 0 10px; font-size: 0.95rem;">Observações por timestamp</h3>
            @if (selectedObservations().length === 0) {
              <div style="color: var(--rv-text-muted); font-size: 0.88rem;">Nenhuma observação registrada.</div>
            } @else {
              <div class="rv-table-wrap">
                <table class="rv-table">
                  <thead>
                    <tr>
                      <th>Frame</th>
                      <th>Timestamp</th>
                      <th>Classe detectada</th>
                      <th>Bounding box (normalizado)</th>
                      <th>Material</th>
                      <th>Nível</th>
                    </tr>
                  </thead>
                  <tbody>
                    @for (obs of selectedObservations(); track obs.id) {
                      <tr class="rv-row--selectable" (click)="seekToTimestamp(obs.frameTimestampMs)">
                        <td>#{{ obs.frameIndex }}</td>
                        <td class="rv-mono">{{ obs.frameTimestampMs }} ms</td>
                        <td class="rv-mono">{{ obs.detectedClass }} ({{ obs.detectionConfidence | number: '1.2-2' }})</td>
                        <td class="rv-mono">
                          [{{ obs.boundingBox.x | number: '1.2-2' }}, {{ obs.boundingBox.y | number: '1.2-2' }},
                          {{ obs.boundingBox.width | number: '1.2-2' }}, {{ obs.boundingBox.height | number: '1.2-2' }}]
                        </td>
                        <td>{{ obs.materialClass | materialLabel }}</td>
                        <td>{{ obs.loadingLevel | loadingLevelLabel }}</td>
                      </tr>
                    }
                  </tbody>
                </table>
              </div>
            }
          </section>
        }
      }
    </div>
  `,
})
export class InspectionDetailPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly route = inject(ActivatedRoute);
  private readonly notifications = inject(NotificationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly id = input<string>();

  @ViewChild('videoEl') private videoRef?: ElementRef<HTMLVideoElement>;
  @ViewChild('overlayCanvas') private canvasRef?: ElementRef<HTMLCanvasElement>;

  protected readonly loading = signal<boolean>(false);
  protected readonly startingProcess = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly inspection = signal<InspectionDetail | null>(null);
  protected readonly activeJob = signal<ProcessingJob | null>(null);
  protected readonly wagons = signal<WagonResult[]>([]);
  protected readonly selectedWagon = signal<WagonResult | null>(null);
  protected readonly selectedObservations = signal<WagonObservation[]>([]);

  private pollSub?: Subscription;

  protected readonly inspectionId = computed(
    () => this.id() ?? this.route.snapshot.paramMap.get('id') ?? ''
  );

  protected readonly mediaUrl = computed(() => {
    const id = this.inspectionId();
    return id ? this.api.mediaUrl(id) : '';
  });

  protected readonly isProcessing = computed(() => {
    const jobStatus = this.activeJob()?.status;
    const inspStatus = this.inspection()?.status;
    return (
      jobStatus === 'QUEUED' ||
      jobStatus === 'RUNNING' ||
      inspStatus === 'QUEUED' ||
      inspStatus === 'PROCESSING'
    );
  });

  protected readonly showDemoBanner = computed(
    () =>
      Boolean(this.activeJob()?.isDemoSimulation) ||
      this.wagons().some((wagon) => wagon.isExample)
  );

  protected readonly activeEngineType = computed(
    () => this.activeJob()?.engineType ?? ''
  );

  protected readonly reviewCount = computed(
    () => this.wagons().filter((wagon) => wagon.reviewRequired).length
  );

  ngOnInit(): void {
    const targetId = this.inspectionId();
    if (!targetId) {
      this.errorMessage.set('Identificador da inspeção ausente.');
      return;
    }
    this.loadInspection(targetId);
  }

  protected loadInspection(targetId: string): void {
    this.loading.set(true);
    this.errorMessage.set(null);

    this.api.getInspection(targetId).subscribe({
      next: (detail) => {
        this.inspection.set(detail);
        this.activeJob.set(detail.latestJob);
        this.loading.set(false);

        if (
          detail.latestJob &&
          (detail.latestJob.status === 'QUEUED' || detail.latestJob.status === 'RUNNING')
        ) {
          this.startJobPolling(detail.latestJob.id, targetId);
        } else if (detail.status === 'COMPLETED' || detail.wagonCount > 0) {
          this.loadWagons(targetId);
        }
      },
      error: (err) => {
        this.loading.set(false);
        this.errorMessage.set(
          extractApiErrorMessage(err, 'Não foi possível carregar os detalhes da inspeção.')
        );
      },
    });
  }

  protected startOrReprocess(): void {
    const targetId = this.inspectionId();
    if (!targetId) {
      return;
    }
    this.startingProcess.set(true);
    this.api.startProcessing(targetId).subscribe({
      next: (job) => {
        this.startingProcess.set(false);
        this.activeJob.set(job);
        this.inspection.update((curr) =>
          curr ? { ...curr, status: 'QUEUED', latestJobId: job.id, latestJob: job } : curr
        );
        this.notifications.info('Processamento enfileirado.');
        this.startJobPolling(job.id, targetId);
      },
      error: (err) => {
        this.startingProcess.set(false);
        this.notifications.error(
          extractApiErrorMessage(err, 'Não foi possível iniciar o processamento.')
        );
      },
    });
  }

  protected selectWagon(wagon: WagonResult): void {
    this.selectedWagon.set(wagon);
    this.api.listObservations(wagon.inspectionId, wagon.id).subscribe({
      next: (observations) => {
        this.selectedObservations.set(observations);
        this.redrawOverlays();
      },
      error: (err) => {
        this.notifications.error(
          extractApiErrorMessage(err, 'Erro ao carregar observações do vagão.')
        );
      },
    });
  }

  protected seekToTimestamp(timestampMs: number): void {
    const video = this.videoRef?.nativeElement;
    if (video) {
      video.currentTime = Math.max(0, timestampMs / 1000);
    }
    this.redrawOverlays();
  }

  protected redrawOverlays(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) {
      return;
    }
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const observations = this.selectedObservations();
    if (observations.length === 0) {
      return;
    }

    const video = this.videoRef?.nativeElement;
    const currentMs = (video?.currentTime ?? 0) * 1000;
    let targetObs = observations[0];
    let bestDiff = Math.abs(targetObs.frameTimestampMs - currentMs);
    for (const obs of observations) {
      const diff = Math.abs(obs.frameTimestampMs - currentMs);
      if (diff < bestDiff) {
        bestDiff = diff;
        targetObs = obs;
      }
    }
    if (video && video.currentTime > 0 && bestDiff > 600) {
      return;
    }

    const box = targetObs.boundingBox;
    const x = box.x * canvas.width;
    const y = box.y * canvas.height;
    const w = box.width * canvas.width;
    const h = box.height * canvas.height;

    ctx.strokeStyle = '#ff6a00';
    ctx.lineWidth = 2.5;
    ctx.strokeRect(x, y, w, h);

    const confText =
      targetObs.detectionConfidence > 0
        ? ` ${(targetObs.detectionConfidence * 100).toFixed(0)}%`
        : '';
    const label = `${targetObs.detectedClass}${confText}${targetObs.isExample ? ' (exemplo)' : ''}`;
    ctx.font = '12px sans-serif';
    const textWidth = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(15, 39, 68, 0.86)';
    ctx.fillRect(x, Math.max(0, y - 22), textWidth + 12, 20);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(label, x + 6, Math.max(14, y - 8));
  }

  protected modelVersionEntries(wagon: WagonResult): [string, string][] {
    return Object.entries(wagon.modelVersions ?? {});
  }

  private loadWagons(targetId: string): void {
    this.api.listWagons(targetId).subscribe({
      next: (wagons) => {
        this.wagons.set(wagons);
        if (wagons.length > 0) {
          this.selectWagon(wagons[0]);
        } else {
          this.selectedWagon.set(null);
          this.selectedObservations.set([]);
        }
      },
      error: (err) => {
        this.notifications.error(
          extractApiErrorMessage(err, 'Erro ao carregar os vagões da inspeção.')
        );
      },
    });
  }

  private startJobPolling(jobId: string, inspectionId: string): void {
    this.pollSub?.unsubscribe();
    this.pollSub = timer(0, 1500)
      .pipe(
        switchMap(() => this.api.getJob(jobId)),
        takeWhile((job) => job.status === 'QUEUED' || job.status === 'RUNNING', true),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe({
        next: (job) => {
          this.activeJob.set(job);
          if (job.status === 'COMPLETED') {
            this.api.getInspection(inspectionId).subscribe((detail) => {
              this.inspection.set(detail);
              this.activeJob.set(detail.latestJob);
            });
            this.loadWagons(inspectionId);
          } else if (job.status === 'FAILED') {
            this.api.getInspection(inspectionId).subscribe((detail) => {
              this.inspection.set(detail);
              this.activeJob.set(detail.latestJob);
            });
          }
        },
        error: (err) => {
          this.notifications.error(
            extractApiErrorMessage(err, 'Erro ao consultar progresso do processamento.')
          );
        },
      });
  }
}
