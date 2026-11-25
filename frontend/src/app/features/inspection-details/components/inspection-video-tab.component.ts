import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  ViewChild,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { MatIconModule } from '@angular/material/icon';
import { InspectionDetail, WagonObservation, WagonResult } from '../../../core/models/inspection.models';
import { computeContainedVideoRect } from './video-overlay.util';

@Component({
  selector: 'rv-inspection-video-tab',
  standalone: true,
  imports: [MatIconModule],
  template: `
    <section class="rv-card">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Player de inspeção com overlay de detecções</h3>
          <p class="rv-card__subtitle">
            Reprodução direta do vídeo original com bounding boxes e trilha do vagão selecionado.
          </p>
        </div>

        @if (selectedObservations().length > 0) {
          <div class="rv-header-actions">
            <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="stepObservation(-1)">
              <mat-icon>chevron_left</mat-icon>
              <span>Anterior</span>
            </button>
            <span class="rv-mono" style="font-size: 0.8rem; color: var(--rv-text-secondary);">
              {{ activeObsIndex() + 1 }} / {{ selectedObservations().length }}
            </span>
            <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="stepObservation(1)">
              <span>Próxima</span>
              <mat-icon>chevron_right</mat-icon>
            </button>
          </div>
        }
      </div>

      <div #videoStage class="rv-video-stage">
        <video
          #videoEl
          controls
          preload="metadata"
          [src]="mediaUrl()"
          (loadedmetadata)="redrawOverlays()"
          (loadeddata)="redrawOverlays()"
          (timeupdate)="redrawOverlays()"
          (seeked)="redrawOverlays()"
          (play)="redrawOverlays()"
        ></video>
        <canvas #overlayCanvas></canvas>
      </div>

      @if (selectedObservations().length > 0) {
        <div style="margin-top: 16px; display: flex; flex-direction: column; gap: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--rv-text-muted);">
            <span>Timeline de observações do vagão selecionado</span>
            <span class="rv-mono">Tempo atual: {{ (currentMs() / 1000).toFixed(2) }} s</span>
          </div>
          <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            @for (obs of selectedObservations().slice(0, 24); track obs.id; let idx = $index) {
              <button
                type="button"
                class="rv-btn rv-btn--secondary rv-btn--sm"
                (click)="seekTo(obs.frameTimestampMs, idx)"
              >
                #{{ obs.frameIndex }} · {{ (obs.frameTimestampMs / 1000).toFixed(1) }}s
              </button>
            }
          </div>
        </div>
      }
    </section>
  `,
})
export class InspectionVideoTabComponent implements AfterViewInit {
  private readonly destroyRef = inject(DestroyRef);

  readonly inspection = input.required<InspectionDetail>();
  readonly mediaUrl = input.required<string>();
  readonly wagons = input<WagonResult[]>([]);
  readonly selectedWagon = input<WagonResult | null>(null);
  readonly selectedObservations = input<WagonObservation[]>([]);
  readonly observationsByWagon = input<Record<string, WagonObservation[]>>({});
  readonly seekTimestampMs = input<number | null>(null);

  readonly seekConsumed = output<void>();

  @ViewChild('videoStage') private videoStageRef?: ElementRef<HTMLDivElement>;
  @ViewChild('videoEl') private videoRef?: ElementRef<HTMLVideoElement>;
  @ViewChild('overlayCanvas') private canvasRef?: ElementRef<HTMLCanvasElement>;

  protected readonly currentMs = signal<number>(0);
  protected readonly activeObsIndex = signal<number>(0);
  private resizeObserver?: ResizeObserver;

  constructor() {
    effect(() => {
      this.selectedObservations();
      this.selectedWagon();
      this.observationsByWagon();
      queueMicrotask(() => this.redrawOverlays());
    });

    effect(() => {
      const ts = this.seekTimestampMs();
      if (ts != null) {
        queueMicrotask(() => {
          this.seekTo(ts, 0);
          this.seekConsumed.emit();
        });
      }
    });
  }

  ngAfterViewInit(): void {
    const stage = this.videoStageRef?.nativeElement;
    if (stage && typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.redrawOverlays());
      this.resizeObserver.observe(stage);
      this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
    }
    this.redrawOverlays();
  }

  seekTo(timestampMs: number, obsIdx = 0): void {
    const video = this.videoRef?.nativeElement;
    if (video) {
      video.currentTime = Math.max(0, timestampMs / 1000);
    }
    this.activeObsIndex.set(obsIdx);
    this.currentMs.set(timestampMs);
    this.redrawOverlays();
  }

  protected stepObservation(delta: number): void {
    const list = this.selectedObservations();
    if (list.length === 0) {
      return;
    }
    const nextIdx = (this.activeObsIndex() + delta + list.length) % list.length;
    this.seekTo(list[nextIdx].frameTimestampMs, nextIdx);
  }

  redrawOverlays(): void {
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
    const detail = this.inspection();
    const contained = computeContainedVideoRect(
      canvas.width,
      canvas.height,
      video?.videoWidth || detail?.widthPx,
      video?.videoHeight || detail?.heightPx
    );
    const ms = (video?.currentTime ?? 0) * 1000;
    this.currentMs.set(ms);

    const pickNearest = (obsList: WagonObservation[]): WagonObservation | null => {
      if (obsList.length === 0) {
        return null;
      }
      let best = obsList[0];
      let bestDiff = Math.abs(best.frameTimestampMs - ms);
      for (const obs of obsList) {
        const diff = Math.abs(obs.frameTimestampMs - ms);
        if (diff < bestDiff) {
          bestDiff = diff;
          best = obs;
        }
      }
      if (video && video.readyState > 0 && bestDiff > 600) {
        return null;
      }
      return best;
    };

    const drawBox = (obs: WagonObservation, primary: boolean, seqIndex?: number): void => {
      const box = obs.boundingBox;
      const x = contained.offsetX + box.x * contained.drawWidth;
      const y = contained.offsetY + box.y * contained.drawHeight;
      const w = box.width * contained.drawWidth;
      const h = box.height * contained.drawHeight;

      ctx.strokeStyle = primary ? '#3b82f6' : 'rgba(34, 184, 207, 0.8)';
      ctx.lineWidth = primary ? 2.5 : 1.5;
      ctx.strokeRect(x, y, w, h);

      const confText =
        obs.detectionConfidence > 0 ? ` ${(obs.detectionConfidence * 100).toFixed(0)}%` : '';
      const prefix = seqIndex != null ? `#${seqIndex} ` : '';
      const label = `${prefix}${obs.detectedClass}${confText}`;
      ctx.font = '600 12px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = primary ? 'rgba(11, 21, 38, 0.92)' : 'rgba(11, 21, 38, 0.68)';
      ctx.fillRect(x, Math.max(0, y - 22), textWidth + 12, 20);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x + 6, Math.max(14, y - 8));
    };

    const selected = this.selectedWagon();
    const byWagon = this.observationsByWagon();
    for (const wagon of this.wagons()) {
      if (selected && wagon.id === selected.id) {
        continue;
      }
      const nearestOther = pickNearest(byWagon[wagon.id] ?? []);
      if (nearestOther) {
        drawBox(nearestOther, false, wagon.wagonIdentity.sequenceIndex);
      }
    }

    const nearestSelected = pickNearest(observations);
    if (nearestSelected) {
      drawBox(nearestSelected, true, selected?.wagonIdentity.sequenceIndex);
    }
  }
}
