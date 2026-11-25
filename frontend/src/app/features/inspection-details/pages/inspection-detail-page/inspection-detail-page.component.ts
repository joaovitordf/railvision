import { Component, DestroyRef, OnInit, computed, inject, input, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { EMPTY, Subject, Subscription, catchError, distinctUntilChanged, map, switchMap, takeWhile, timer } from 'rxjs';
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
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { InspectionObservationsTabComponent } from '../../components/inspection-observations-tab.component';
import { InspectionReportTabComponent } from '../../components/inspection-report-tab.component';
import { InspectionSummaryTabComponent } from '../../components/inspection-summary-tab.component';
import { InspectionVideoTabComponent } from '../../components/inspection-video-tab.component';
import { InspectionWagonsTabComponent } from '../../components/inspection-wagons-tab.component';

export { computeContainedVideoRect } from '../../components/video-overlay.util';
export type { ContainedVideoRect } from '../../components/video-overlay.util';

export type InspectionDetailTab = 'summary' | 'video' | 'wagons' | 'observations' | 'report';

@Component({
  selector: 'rv-inspection-detail-page',
  standalone: true,
  imports: [
    RouterLink,
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    DemoFixtureBannerComponent,
    InspectionSummaryTabComponent,
    InspectionVideoTabComponent,
    InspectionWagonsTabComponent,
    InspectionObservationsTabComponent,
    InspectionReportTabComponent,
  ],
  template: `
    <div class="rv-page">
      <div class="rv-page-header">
        <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
          <a class="rv-btn rv-btn--secondary" routerLink="/inspections">
            <mat-icon>arrow_back</mat-icon>
            <span>Inspeções</span>
          </a>
          @if (inspection(); as detail) {
            <div>
              <h2 class="rv-page-header__title" style="font-size: 1.3rem;">{{ detail.title }}</h2>
              <div class="rv-mono" style="font-size: 0.78rem; color: var(--rv-text-muted); margin-top: 2px;">
                {{ detail.originalFilename }}
              </div>
            </div>
          }
        </div>

        @if (inspection(); as detail) {
          <div class="rv-header-actions">
            <rv-status-badge [status]="detail.status" />
            <button
              type="button"
              class="rv-btn rv-btn--primary"
              [disabled]="isProcessing() || startingProcess()"
              (click)="startOrReprocess()"
            >
              <mat-icon>play_arrow</mat-icon>
              <span>{{ detail.status === 'UPLOADED' ? 'Iniciar processamento' : 'Reprocessar' }}</span>
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
                  <h3 class="rv-card__title">Processamento em andamento</h3>
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
              <div style="display: flex; flex-direction: column; gap: 4px;">
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

        <nav class="rv-tabs" role="tablist" aria-label="Seções da inspeção">
          <button
            type="button"
            role="tab"
            class="rv-tab"
            data-testid="tab-summary"
            [class.rv-tab--active]="activeTab() === 'summary'"
            (click)="activeTab.set('summary')"
          >
            <mat-icon>dashboard</mat-icon>
            <span>Resumo</span>
          </button>
          <button
            type="button"
            role="tab"
            class="rv-tab"
            data-testid="tab-video"
            [class.rv-tab--active]="activeTab() === 'video'"
            (click)="activeTab.set('video')"
          >
            <mat-icon>smart_display</mat-icon>
            <span>Vídeo</span>
          </button>
          <button
            type="button"
            role="tab"
            class="rv-tab"
            data-testid="tab-wagons"
            [class.rv-tab--active]="activeTab() === 'wagons'"
            (click)="activeTab.set('wagons')"
          >
            <mat-icon>view_carousel</mat-icon>
            <span>Vagões</span>
            <span class="rv-badge rv-badge--neutral">{{ wagons().length }}</span>
          </button>
          <button
            type="button"
            role="tab"
            class="rv-tab"
            data-testid="tab-observations"
            [class.rv-tab--active]="activeTab() === 'observations'"
            (click)="activeTab.set('observations')"
          >
            <mat-icon>visibility</mat-icon>
            <span>Observações</span>
            <span class="rv-badge rv-badge--neutral">{{ selectedObservations().length }}</span>
          </button>
          <button
            type="button"
            role="tab"
            class="rv-tab"
            data-testid="tab-report"
            [class.rv-tab--active]="activeTab() === 'report'"
            (click)="activeTab.set('report')"
          >
            <mat-icon>summarize</mat-icon>
            <span>Relatório</span>
          </button>
        </nav>

        @switch (activeTab()) {
          @case ('summary') {
            <rv-inspection-summary-tab
              [detail]="detail"
              [latestJob]="activeJob()"
              [wagons]="wagons()"
              [reviewCount]="reviewCount()"
            />
          }
          @case ('video') {
            <rv-inspection-video-tab
              [inspection]="detail"
              [mediaUrl]="mediaUrl()"
              [wagons]="wagons()"
              [selectedWagon]="selectedWagon()"
              [selectedObservations]="selectedObservations()"
              [observationsByWagon]="observationsByWagon()"
              [seekTimestampMs]="pendingSeekMs()"
              (seekConsumed)="pendingSeekMs.set(null)"
            />
          }
          @case ('wagons') {
            <rv-inspection-wagons-tab
              [inspectionStatus]="detail.status"
              [wagons]="wagons()"
              [selectedWagon]="selectedWagon()"
              (wagonSelected)="selectWagon($event)"
            />
          }
          @case ('observations') {
            <rv-inspection-observations-tab
              [selectedWagon]="selectedWagon()"
              [observations]="selectedObservations()"
              (openInVideo)="jumpToVideoTimestamp($event)"
            />
          }
          @case ('report') {
            <rv-inspection-report-tab
              [detail]="detail"
              [wagons]="wagons()"
              [observationsByWagon]="observationsByWagon()"
            />
          }
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

  protected readonly activeTab = signal<InspectionDetailTab>('summary');
  protected readonly pendingSeekMs = signal<number | null>(null);
  protected readonly loading = signal<boolean>(false);
  protected readonly startingProcess = signal<boolean>(false);
  protected readonly errorMessage = signal<string | null>(null);
  protected readonly inspection = signal<InspectionDetail | null>(null);
  protected readonly activeJob = signal<ProcessingJob | null>(null);
  protected readonly wagons = signal<WagonResult[]>([]);
  protected readonly selectedWagon = signal<WagonResult | null>(null);
  protected readonly selectedObservations = signal<WagonObservation[]>([]);
  protected readonly observationsByWagon = signal<Record<string, WagonObservation[]>>({});

  private readonly routeParamId = signal<string>('');
  private readonly wagonSelect$ = new Subject<WagonResult>();
  private pollSub?: Subscription;

  protected readonly inspectionId = computed(
    () => this.id() || this.routeParamId() || this.route.snapshot?.paramMap?.get('id') || ''
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

  protected readonly activeEngineType = computed(() => this.activeJob()?.engineType ?? '');
  protected readonly reviewCount = computed(
    () => this.wagons().filter((wagon) => wagon.reviewRequired).length
  );

  ngOnInit(): void {
    const initialTab = this.route.snapshot?.data?.['initialTab'] as InspectionDetailTab | undefined;
    if (initialTab) {
      this.activeTab.set(initialTab);
    }

    this.wagonSelect$
      .pipe(
        switchMap((wagon) => {
          const cached = this.observationsByWagon()[wagon.id];
          if (cached) {
            return EMPTY;
          }
          return this.api.listObservations(wagon.inspectionId, wagon.id).pipe(
            catchError((err) => {
              this.notifications.error(
                extractApiErrorMessage(err, 'Erro ao carregar observações do vagão.')
              );
              return EMPTY;
            }),
            map((observations) => ({ wagonId: wagon.id, observations }))
          );
        }),
        takeUntilDestroyed(this.destroyRef)
      )
      .subscribe(({ wagonId, observations }) => {
        this.observationsByWagon.update((curr) => ({ ...curr, [wagonId]: observations }));
        if (this.selectedWagon()?.id === wagonId) {
          this.selectedObservations.set(observations);
        }
      });

    if (this.route.paramMap) {
      this.route.paramMap
        .pipe(
          map((params) => this.id() || params.get('id') || ''),
          distinctUntilChanged(),
          takeUntilDestroyed(this.destroyRef)
        )
        .subscribe((targetId) => {
          this.routeParamId.set(targetId);
          if (!targetId) {
            this.errorMessage.set('Identificador da inspeção ausente.');
            return;
          }
          this.stopJobPolling();
          this.wagons.set([]);
          this.selectedWagon.set(null);
          this.selectedObservations.set([]);
          this.observationsByWagon.set({});
          this.loadInspection(targetId);
        });
      return;
    }

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
    this.wagons.set([]);
    this.selectedWagon.set(null);
    this.selectedObservations.set([]);
    this.observationsByWagon.set({});
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
    const cached = this.observationsByWagon()[wagon.id];
    if (cached) {
      this.selectedObservations.set(cached);
      return;
    }
    this.selectedObservations.set([]);
    this.wagonSelect$.next(wagon);
  }

  protected jumpToVideoTimestamp(timestampMs: number): void {
    this.pendingSeekMs.set(timestampMs);
    this.activeTab.set('video');
  }

  private stopJobPolling(): void {
    this.pollSub?.unsubscribe();
    this.pollSub = undefined;
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
