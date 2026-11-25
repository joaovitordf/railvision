import { Component, DestroyRef, OnInit, computed, inject, input, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ActivatedRoute } from '@angular/router';
import { Subscription, catchError, distinctUntilChanged, map, of, switchMap, takeWhile, timer } from 'rxjs';
import { ModelVersion } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { LabComparePanelComponent } from '../../components/lab-compare-panel.component';
import { LabHistoryTableComponent } from '../../components/lab-history-table.component';
import { LabModelsRegistryComponent } from '../../components/lab-models-registry.component';
import { LabRunFormComponent } from '../../components/lab-run-form.component';
import { LabRunViewerComponent } from '../../components/lab-run-viewer.component';
import { LabTrainingPanelComponent } from '../../components/lab-training-panel.component';
import {
  CurateFrameRequest,
  CurationExport,
  LabRunResult,
  ModelCheckpoint,
  RunComparison,
  RunListItem,
  RuntimeStatus,
  VideoRunStatus,
} from '../../models/lab.models';
import { LabApiService } from '../../services/lab-api.service';

const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'webp']);
const VIDEO_EXTS = new Set(['mp4', 'webm', 'ogv', 'mov', 'mkv', 'avi']);

export type LabTab = 'inference' | 'compare' | 'history' | 'models' | 'training';

@Component({
  selector: 'rv-lab-page',
  standalone: true,
  imports: [
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    LabRunFormComponent,
    LabRunViewerComponent,
    LabComparePanelComponent,
    LabHistoryTableComponent,
    LabModelsRegistryComponent,
    LabTrainingPanelComponent,
  ],
  template: `
    <div class="rv-page">
      <div class="rv-page-header">
        <div>
          <h2 class="rv-page-header__title">Laboratório de Visão Computacional</h2>
          <p class="rv-page-header__subtitle">
            Experimentação local de checkpoints, comparação A/B e gestão do Model Registry (sem alterar datasets congelados).
          </p>
        </div>

        <div class="rv-header-actions">
          @if (status(); as st) {
            <span
              class="rv-badge"
              [class.rv-badge--success]="hasAnyAvailableRuntime()"
              [class.rv-badge--error]="!hasAnyAvailableRuntime()"
            >
              <span class="rv-badge__dot" aria-hidden="true"></span>
              {{ st.available ? st.deviceType + ' · ' + st.activeExecutionProvider : 'Runtime indisponível' }}
            </span>
          }
          <button type="button" class="rv-btn rv-btn--secondary" (click)="refreshRuns()">
            <mat-icon>refresh</mat-icon>
            <span>Atualizar</span>
          </button>
        </div>
      </div>

      @if (errorMessage()) {
        <div class="rv-banner rv-banner--error" role="alert">
          <mat-icon aria-hidden="true">error_outline</mat-icon>
          <div>{{ errorMessage() }}</div>
        </div>
      }

      @if (loadingStatus()) {
        <mat-progress-bar mode="indeterminate" />
      }

      <nav class="rv-tabs" role="tablist" aria-label="Áreas do Laboratório">
        <button
          type="button"
          role="tab"
          class="rv-tab"
          data-testid="lab-tab-inference"
          [class.rv-tab--active]="activeTab() === 'inference'"
          (click)="activeTab.set('inference')"
        >
          <mat-icon>science</mat-icon>
          <span>Inferência</span>
        </button>
        <button
          type="button"
          role="tab"
          class="rv-tab"
          data-testid="lab-tab-compare"
          [class.rv-tab--active]="activeTab() === 'compare'"
          (click)="activeTab.set('compare')"
        >
          <mat-icon>compare_arrows</mat-icon>
          <span>Comparação A/B</span>
        </button>
        <button
          type="button"
          role="tab"
          class="rv-tab"
          data-testid="lab-tab-history"
          [class.rv-tab--active]="activeTab() === 'history'"
          (click)="activeTab.set('history')"
        >
          <mat-icon>history</mat-icon>
          <span>Histórico ({{ runs().length }})</span>
        </button>
        <button
          type="button"
          role="tab"
          class="rv-tab"
          data-testid="lab-tab-models"
          [class.rv-tab--active]="activeTab() === 'models'"
          (click)="activeTab.set('models')"
        >
          <mat-icon>memory</mat-icon>
          <span>Modelos & Runtime ({{ checkpoints().length }})</span>
        </button>
        <button
          type="button"
          role="tab"
          class="rv-tab"
          data-testid="lab-tab-training"
          [class.rv-tab--active]="activeTab() === 'training'"
          (click)="activeTab.set('training')"
        >
          <mat-icon>trending_up</mat-icon>
          <span>Treinamento</span>
        </button>
      </nav>

      @switch (activeTab()) {
        @case ('inference') {
          @if (status(); as st) {
            @if (!st.available && hasAnyAvailableRuntime()) {
              <div class="rv-banner rv-banner--warning" role="status">
                <mat-icon aria-hidden="true">warning</mat-icon>
                <div>{{ st.unavailableReason }} Selecione um runtime CPU disponível ou habilite o fallback explícito.</div>
              </div>
            } @else if (!hasAnyAvailableRuntime()) {
              <div class="rv-banner rv-banner--error" role="alert">
                <mat-icon aria-hidden="true">warning</mat-icon>
                <div>{{ st.unavailableReason }}</div>
              </div>
            }
          }

          <rv-lab-run-form
            [status]="status()"
            [checkpoints]="checkpoints()"
            [selectedModelId]="selectedModelId()"
            [selectedFile]="selectedFile()"
            [selectedMediaKind]="selectedMediaKind()"
            [confThreshold]="confThreshold()"
            [iouThreshold]="iouThreshold()"
            [selectedRuntime]="selectedRuntime()"
            [sampleFps]="sampleFps()"
            [outputResolution]="outputResolution()"
            [enableTracking]="enableTracking()"
            [allowCpuFallback]="allowCpuFallback()"
            [submitting]="submitting()"
            [canSubmit]="canSubmit()"
            (fileSelected)="onFileSelected($event)"
            (modelChanged)="onModelSelected($event)"
            (confDraftChanged)="onNumericDraft($event, confThreshold)"
            (iouDraftChanged)="onNumericDraft($event, iouThreshold)"
            (thresholdsBlurred)="clampThresholds()"
            (runtimeChanged)="selectedRuntime.set($event)"
            (sampleFpsChanged)="sampleFps.set($event)"
            (outputResolutionChanged)="outputResolution.set($event)"
            (trackingChanged)="enableTracking.set($event)"
            (allowCpuFallbackChanged)="allowCpuFallback.set($event)"
            (submitRun)="runAnalysis()"
          />

          @if (activeVideoProgress(); as vp) {
            <section class="rv-card" data-testid="video-progress-card">
              <div class="rv-card__header">
                <div>
                  <h3 class="rv-card__title">Processando vídeo em Python + PyAV...</h3>
                  <p class="rv-card__subtitle">
                    <span class="rv-mono">{{ vp.originalFilename }}</span> · Frames processados:
                    <strong>{{ vp.processedFrames }} / {{ vp.estimatedTotalFrames }}</strong> ({{ vp.progressPercent }}%)
                  </p>
                </div>
                <rv-status-badge [status]="vp.status" />
              </div>
              <mat-progress-bar
                [mode]="vp.progressPercent > 0 ? 'determinate' : 'indeterminate'"
                [value]="vp.progressPercent"
              />
            </section>
          }

          <rv-lab-run-viewer
            [activeRun]="activeRun()"
            [showOriginalImage]="showOriginalImage()"
            [onlyDetectedFrames]="onlyDetectedFrames()"
            [curating]="curating()"
            [lastCuration]="lastCuration()"
            (toggleOriginalImage)="showOriginalImage.set(!showOriginalImage())"
            (toggleOnlyDetected)="onlyDetectedFrames.set(!onlyDetectedFrames())"
            (curateRequested)="handleCurate($event.runId, $event.payload)"
          />
        }

        @case ('compare') {
          <rv-lab-compare-panel
            [runs]="runs()"
            [comparison]="comparison()"
            [comparing]="comparing()"
            (compareRequested)="handleCompare($event.runA, $event.runB)"
          />
        }

        @case ('history') {
          <rv-lab-history-table
            [runs]="runs()"
            [activeRunId]="activeRun()?.runId ?? null"
            [runsError]="runsError()"
            (refreshRequested)="refreshRuns()"
            (runSelected)="selectRunAndOpen($event)"
          />
        }

        @case ('models') {
          <rv-lab-models-registry
            [status]="status()"
            [checkpoints]="checkpoints()"
            [dbModels]="dbModels()"
            (selectForInference)="testCheckpoint($event)"
          />
        }

        @case ('training') {
          <rv-lab-training-panel
            (checkpointRegistered)="reloadCheckpoints()"
            (testRegisteredModel)="onTestTrainedModel($event)"
          />
        }
      }
    </div>
  `,
})
export class LabPageComponent implements OnInit {
  private readonly api = inject(LabApiService);
  private readonly coreApi = inject(RailvisionApiService, { optional: true });
  private readonly notifications = inject(NotificationService);
  private readonly route = inject(ActivatedRoute, { optional: true });
  private readonly destroyRef = inject(DestroyRef);

  readonly runId = input<string>();

  readonly activeTab = signal<LabTab>('inference');
  readonly loadingStatus = signal(true);
  readonly status = signal<RuntimeStatus | null>(null);
  readonly checkpoints = signal<ModelCheckpoint[]>([]);
  readonly dbModels = signal<ModelVersion[]>([]);
  readonly selectedModelId = signal<string>('exp04c_yolo26n_opentop_640');
  readonly selectedFile = signal<File | null>(null);
  readonly confThreshold = signal(0.25);
  readonly iouThreshold = signal(0.45);
  readonly sampleFps = signal<number>(5);
  readonly selectedRuntime = signal<string>('pytorch_cuda');
  readonly outputResolution = signal<string>('1080p');
  readonly enableTracking = signal<boolean>(false);
  readonly allowCpuFallback = signal<boolean>(false);
  readonly submitting = signal(false);
  readonly errorMessage = signal<string | null>(null);
  readonly runsError = signal<string | null>(null);
  readonly activeRun = signal<LabRunResult | null>(null);
  readonly runs = signal<RunListItem[]>([]);
  readonly showOriginalImage = signal(false);
  readonly onlyDetectedFrames = signal(false);
  readonly comparing = signal(false);
  readonly comparison = signal<RunComparison | null>(null);
  readonly curating = signal(false);
  readonly lastCuration = signal<CurationExport | null>(null);

  private pollSubscription: Subscription | null = null;

  readonly hasAnyAvailableRuntime = computed(() => {
    const st = this.status();
    if (!st) {
      return false;
    }
    return Boolean(st.available || st.availableRuntimes?.some((opt) => opt.available));
  });

  readonly selectedRuntimeAvailable = computed(() => {
    const st = this.status();
    if (!st) {
      return false;
    }
    if (!st.availableRuntimes?.length) {
      return st.available;
    }
    const chosen = this.selectedRuntime();
    const match = st.availableRuntimes.find((opt) => opt.id === chosen);
    if (!match) {
      return st.available;
    }
    return Boolean(match.available || (chosen === 'pytorch_cuda' && this.allowCpuFallback()));
  });

  readonly selectedMediaKind = computed<'IMAGE' | 'VIDEO' | 'UNKNOWN' | null>(() => {
    const file = this.selectedFile();
    if (!file) {
      return null;
    }
    const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
    if (IMAGE_EXTS.has(ext) || file.type.startsWith('image/')) {
      return 'IMAGE';
    }
    if (VIDEO_EXTS.has(ext) || file.type.startsWith('video/')) {
      return 'VIDEO';
    }
    return 'UNKNOWN';
  });

  readonly canSubmit = computed(() => {
    const kind = this.selectedMediaKind();
    return Boolean(
      this.hasAnyAvailableRuntime() &&
        this.selectedRuntimeAvailable() &&
        !this.submitting() &&
        (kind === 'IMAGE' || kind === 'VIDEO')
    );
  });

  readonly activeVideoProgress = computed<VideoRunStatus | null>(() => {
    const run = this.activeRun();
    if (run && run.mediaType === 'VIDEO' && (run.status === 'QUEUED' || run.status === 'RUNNING')) {
      return run;
    }
    return null;
  });

  constructor() {
    this.destroyRef.onDestroy(() => this.stopPolling());
  }

  ngOnInit(): void {
    const initialTab = this.route?.snapshot?.data?.['initialTab'] as LabTab | undefined;
    if (initialTab) {
      this.activeTab.set(initialTab);
    }

    this.loadStatus();
    if (typeof this.api.listCheckpoints === 'function') {
      this.api
        .listCheckpoints()
        .pipe(catchError(() => of([] as ModelCheckpoint[])), takeUntilDestroyed(this.destroyRef))
        .subscribe((cps) => {
          this.checkpoints.set(cps);
          const active = cps.find((c) => c.status === 'active') ?? cps[0];
          if (active) {
            this.selectedModelId.set(active.id);
          }
        });
    }
    if (this.coreApi && typeof this.coreApi.listModels === 'function') {
      this.coreApi
        .listModels()
        .pipe(catchError(() => of([] as ModelVersion[])), takeUntilDestroyed(this.destroyRef))
        .subscribe((models) => this.dbModels.set(models));
    }
    this.refreshRuns();

    if (this.route?.paramMap) {
      this.route.paramMap
        .pipe(
          map((p) => this.runId() || p.get('runId') || ''),
          distinctUntilChanged(),
          takeUntilDestroyed(this.destroyRef)
        )
        .subscribe((targetRunId) => {
          if (targetRunId) {
            this.selectRunAndOpen(targetRunId);
          }
        });
    } else if (this.runId()) {
      this.selectRunAndOpen(this.runId()!);
    }
  }

  onModelSelected(modelId: string): void {
    this.selectedModelId.set(modelId);
    this.loadStatus(modelId);
  }

  testCheckpoint(modelId: string): void {
    this.onModelSelected(modelId);
    this.activeTab.set('inference');
  }

  reloadCheckpoints(): void {
    if (typeof this.api.listCheckpoints !== 'function') {
      return;
    }
    this.api
      .listCheckpoints()
      .pipe(catchError(() => of([] as ModelCheckpoint[])), takeUntilDestroyed(this.destroyRef))
      .subscribe((cps) => this.checkpoints.set(cps));
  }

  onTestTrainedModel(modelId: string): void {
    this.reloadCheckpoints();
    this.testCheckpoint(modelId);
  }

  private loadStatus(modelId?: string): void {
    this.api
      .getStatus(modelId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (st) => {
          this.status.set(st);
          if (modelId == null) {
            this.confThreshold.set(st.defaultConfThreshold);
            this.iouThreshold.set(st.defaultIouThreshold);
            if (st.defaultOutputResolution) {
              this.outputResolution.set(st.defaultOutputResolution);
            }
          }
          if (st.activeModelId) {
            this.selectedModelId.set(st.activeModelId);
          }
          const runtimes = st.availableRuntimes ?? [];
          const currentSelected = runtimes.find((opt) => opt.id === this.selectedRuntime() && opt.available);
          if (!currentSelected) {
            const defaultOpt = runtimes.find((opt) => opt.id === st.defaultRuntime && opt.available);
            if (defaultOpt) {
              this.selectedRuntime.set(defaultOpt.id);
            } else {
              const firstAvailable = runtimes.find((opt) => opt.available);
              if (firstAvailable) {
                this.selectedRuntime.set(firstAvailable.id);
              } else if (st.defaultRuntime) {
                this.selectedRuntime.set(st.defaultRuntime);
              }
            }
          }
          this.loadingStatus.set(false);
        },
        error: (err) => {
          this.loadingStatus.set(false);
          this.errorMessage.set(
            extractApiErrorMessage(err, 'Falha ao consultar o status do runtime de visão.')
          );
        },
      });
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files && input.files.length > 0 ? input.files[0] : null;
    this.selectedFile.set(file);
    this.errorMessage.set(null);
    input.value = '';
  }

  onNumericDraft(raw: unknown, targetSignal: { set: (val: number) => void }): void {
    const parsed = Number(raw);
    if (Number.isFinite(parsed)) {
      targetSignal.set(parsed);
    }
  }

  clampThresholds(): void {
    this.confThreshold.set(this.toBoundedNumber(this.confThreshold(), 0.25, 0.01, 0.99));
    this.iouThreshold.set(this.toBoundedNumber(this.iouThreshold(), 0.45, 0.05, 0.95));
  }

  runAnalysis(): void {
    const file = this.selectedFile();
    const kind = this.selectedMediaKind();
    if (!file || (kind !== 'IMAGE' && kind !== 'VIDEO')) {
      this.errorMessage.set('Selecione uma imagem ou vídeo em formato suportado.');
      return;
    }

    this.clampThresholds();
    this.submitting.set(true);
    this.errorMessage.set(null);
    this.showOriginalImage.set(false);
    this.lastCuration.set(null);
    this.stopPolling();

    const st = this.status();
    const runtimeArg = st?.availableRuntimes?.length ? this.selectedRuntime() : null;
    const fallbackArg = this.allowCpuFallback();
    const modelArg = this.checkpoints().length ? this.selectedModelId() : null;

    if (kind === 'IMAGE') {
      this.api
        .analyzeImage(file, this.confThreshold(), this.iouThreshold(), runtimeArg, fallbackArg, modelArg)
        .pipe(takeUntilDestroyed(this.destroyRef))
        .subscribe({
          next: (res) => {
            this.submitting.set(false);
            this.activeRun.set(res);
            this.refreshRuns();
            this.notifications.success('Imagem processada com sucesso no laboratório.');
          },
          error: (err) => {
            this.submitting.set(false);
            const msg = extractApiErrorMessage(err, 'Falha ao analisar imagem no laboratório.');
            this.errorMessage.set(msg);
            this.notifications.error(msg);
          },
        });
      return;
    }

    const fpsArg = this.sampleFps() > 0 ? this.sampleFps() : null;
    const outResArg = st?.availableOutputResolutions?.length ? this.outputResolution() : null;
    const trackingArg = Boolean(st?.trackingAlgorithms?.includes('bytetrack') && this.enableTracking());
    this.api
      .analyzeVideo(
        file,
        this.confThreshold(),
        this.iouThreshold(),
        fpsArg,
        runtimeArg,
        outResArg,
        trackingArg,
        fallbackArg,
        modelArg
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (initial) => {
          this.submitting.set(false);
          this.activeRun.set(initial);
          this.refreshRuns();
          this.startPolling(initial.runId);
        },
        error: (err) => {
          this.submitting.set(false);
          const msg = extractApiErrorMessage(err, 'Falha ao enviar vídeo para análise no laboratório.');
          this.errorMessage.set(msg);
          this.notifications.error(msg);
        },
      });
  }

  selectRunAndOpen(runId: string): void {
    this.activeTab.set('inference');
    this.selectRun(runId);
  }

  selectRun(runId: string): void {
    this.stopPolling();
    this.errorMessage.set(null);
    this.lastCuration.set(null);
    this.api
      .getRun(runId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (run) => {
          this.activeRun.set(run);
          if (run.mediaType === 'VIDEO' && (run.status === 'QUEUED' || run.status === 'RUNNING')) {
            this.startPolling(run.runId);
          }
        },
        error: (err) =>
          this.errorMessage.set(
            extractApiErrorMessage(err, 'Falha ao carregar detalhes da execução selecionada.')
          ),
      });
  }

  refreshRuns(): void {
    this.runsError.set(null);
    this.api
      .listRuns()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.runs.set(list),
        error: (err) => {
          this.runsError.set(
            extractApiErrorMessage(err, 'Não foi possível carregar o histórico de execuções.')
          );
        },
      });
  }

  handleCompare(runA: string, runB: string): void {
    if (typeof this.api.compareRuns !== 'function') {
      return;
    }
    this.comparing.set(true);
    this.api
      .compareRuns(runA, runB)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.comparison.set(res);
          this.comparing.set(false);
        },
        error: (err) => {
          this.comparing.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao comparar execuções.'));
        },
      });
  }

  handleCurate(runId: string, payload: CurateFrameRequest): void {
    if (typeof this.api.curateFrame !== 'function') {
      return;
    }
    this.curating.set(true);
    this.api
      .curateFrame(runId, payload)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (res) => {
          this.curating.set(false);
          this.lastCuration.set(res);
          this.notifications.success(`Frame #${res.frameIndex} exportado para curadoria.`);
        },
        error: (err) => {
          this.curating.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao enviar frame para curadoria.'));
        },
      });
  }

  toBoundedNumber(raw: unknown, fallback: number, min: number, max: number): number {
    const val = Number(raw);
    if (!Number.isFinite(val)) {
      return fallback;
    }
    return Math.max(min, Math.min(max, val));
  }

  private startPolling(runId: string): void {
    this.stopPolling();
    this.pollSubscription = timer(800, 1000)
      .pipe(
        switchMap(() => this.api.getRun(runId)),
        takeWhile(
          (run) => run.mediaType === 'VIDEO' && (run.status === 'QUEUED' || run.status === 'RUNNING'),
          true
        ),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe({
        next: (run) => {
          this.activeRun.set(run);
          if (run.mediaType === 'VIDEO' && run.status === 'COMPLETED') {
            this.refreshRuns();
            this.notifications.success('Vídeo anotado pronto no Laboratório.');
          } else if (run.mediaType === 'VIDEO' && run.status === 'FAILED') {
            this.refreshRuns();
            this.notifications.error(run.errorMessage || 'Falha no processamento do vídeo.');
          }
        },
        error: (err) => {
          this.errorMessage.set(
            extractApiErrorMessage(err, 'Falha ao acompanhar o processamento do vídeo.')
          );
          this.stopPolling();
        },
      });
  }

  private stopPolling(): void {
    this.pollSubscription?.unsubscribe();
    this.pollSubscription = null;
  }
}
