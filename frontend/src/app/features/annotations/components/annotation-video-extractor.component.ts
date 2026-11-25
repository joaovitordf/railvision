import { Component, computed, effect, inject, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { extractApiErrorMessage } from '../../../core/services/api-error.util';
import { NotificationService } from '../../../core/services/notification.service';
import { FormatDurationPipe } from '../../../shared/pipes/format-duration.pipe';
import { AnnotationVideo, PreviewFrameCandidate } from '../models/annotations.models';
import { AnnotationsApiService } from '../services/annotations-api.service';

@Component({
  selector: 'rv-annotation-video-extractor',
  standalone: true,
  imports: [
    FormsModule,
    MatIconModule,
    MatProgressBarModule,
    FormatDurationPipe,
  ],
  template: `
    <section class="rv-card">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Extração inteligente de frames de vídeo</h3>
          <p class="rv-card__subtitle">
            Envie um vídeo, gere miniaturas com deduplicação visual e importe apenas os frames variados necessários.
          </p>
        </div>
        <label class="rv-btn rv-btn--secondary" [style.opacity]="uploadingVideo() ? '0.6' : '1'">
          <mat-icon>movie</mat-icon>
          <span>{{ uploadingVideo() ? 'Enviando...' : 'Importar vídeo para extração' }}</span>
          <input
            type="file"
            accept=".mp4,.mov,.mkv,.webm,.avi,.ogv,video/*"
            [disabled]="uploadingVideo()"
            (change)="onVideoUpload($event)"
            style="display: none;"
          />
        </label>
      </div>

      @if (uploadingVideo()) {
        <mat-progress-bar mode="indeterminate" style="margin-bottom: 12px;" />
      }

      @if (videos().length === 0) {
        <div class="rv-dropzone" (click)="emptyVideoInput.click()">
          <div class="rv-dropzone__left">
            <div class="rv-dropzone__icon">
              <mat-icon aria-hidden="true">movie</mat-icon>
            </div>
            <div>
              <div class="rv-dropzone__title">Nenhum vídeo enviado neste projeto ainda</div>
              <div class="rv-dropzone__hint">
                Clique para enviar MP4, WEBM, OGV, MOV, MKV ou AVI e extrair frames variados sem duplicatas.
              </div>
            </div>
          </div>
          <input
            #emptyVideoInput
            type="file"
            accept=".mp4,.mov,.mkv,.webm,.avi,.ogv,video/*"
            [disabled]="uploadingVideo()"
            (change)="onVideoUpload($event)"
            style="display: none;"
          />
          <span class="rv-btn rv-btn--secondary rv-btn--sm">Selecionar vídeo</span>
        </div>
      } @else {
        <div class="rv-form-grid" style="margin-bottom: 14px;">
          <div class="rv-field">
            <label class="rv-label" for="ext-vid-select">Vídeo de origem</label>
            <select
              id="ext-vid-select"
              class="rv-select"
              [ngModel]="activeVideoId()"
              (ngModelChange)="activeVideoId.set($event)"
            >
              @for (v of videos(); track v.id) {
                <option [ngValue]="v.id">
                  {{ v.filename }} ({{ v.widthPx }}×{{ v.heightPx }} · {{ v.durationMs | formatDuration }})
                </option>
              }
            </select>
          </div>

          <div class="rv-field">
            <label class="rv-label" for="ext-fps-select">Taxa (FPS de varredura)</label>
            <select
              id="ext-fps-select"
              class="rv-select"
              [ngModel]="sampleFps()"
              (ngModelChange)="sampleFps.set($event)"
            >
              <option [ngValue]="0.5">0,5 fps (1 frame a cada 2s)</option>
              <option [ngValue]="1">1 fps</option>
              <option [ngValue]="2">2 fps</option>
              <option [ngValue]="5">5 fps</option>
            </select>
          </div>

          <div class="rv-field">
            <label class="rv-label" for="ext-max-input">Máximo de frames</label>
            <input
              id="ext-max-input"
              class="rv-input"
              type="number"
              min="1"
              max="120"
              [ngModel]="maxFrames()"
              (ngModelChange)="maxFrames.set(+$event)"
            />
          </div>

          <div class="rv-field">
            <label class="rv-label" for="ext-dedup-input">Filtro anti-duplicata (diff mín.)</label>
            <input
              id="ext-dedup-input"
              class="rv-input"
              type="number"
              min="0"
              max="40"
              step="0.5"
              [ngModel]="dedupThreshold()"
              (ngModelChange)="dedupThreshold.set(+$event)"
            />
          </div>

          <div style="display: flex;">
            <button
              type="button"
              class="rv-btn rv-btn--primary"
              [disabled]="!activeVideoId() || previewing()"
              (click)="generatePreview()"
            >
              <mat-icon>burst_mode</mat-icon>
              <span>{{ previewing() ? 'Analisando...' : 'Gerar miniaturas' }}</span>
            </button>
          </div>
        </div>

        @if (previewing() || extracting()) {
          <mat-progress-bar mode="indeterminate" style="margin-bottom: 12px;" />
        }

        @if (candidates().length > 0) {
          <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 0.86rem; color: var(--rv-text-secondary);">
              <strong>{{ selectedCount() }}</strong> de {{ candidates().length }} candidatos selecionados
              @if (skippedSimilar() > 0) {
                · {{ skippedSimilar() }} frames quase idênticos ignorados automaticamente
              }
            </span>

            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
              <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="selectAll(true)">
                Selecionar todos
              </button>
              <button type="button" class="rv-btn rv-btn--secondary rv-btn--sm" (click)="selectAll(false)">
                Limpar seleção
              </button>
              <button
                type="button"
                class="rv-btn rv-btn--primary rv-btn--sm"
                [disabled]="selectedCount() === 0 || extracting()"
                (click)="commitSelected()"
              >
                <mat-icon>save_alt</mat-icon>
                <span>{{ extracting() ? 'Importando...' : 'Importar selecionados (' + selectedCount() + ')' }}</span>
              </button>
            </div>
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(175px, 1fr)); gap: 12px; max-height: 360px; overflow-y: auto;">
            @for (cand of candidates(); track cand.frameIndex) {
              <div
                (click)="toggleCandidate(cand.frameIndex)"
                style="position: relative; border: 2px solid; border-radius: var(--rv-radius-sm); overflow: hidden; cursor: pointer; background: var(--rv-bg-surface-muted); transition: border-color 120ms ease;"
                [style.border-color]="selectedIndices().has(cand.frameIndex) ? 'var(--rv-accent)' : 'var(--rv-border)'"
              >
                <img
                  [src]="cand.previewUrl"
                  alt="Frame candidato"
                  style="width: 100%; height: 100px; object-fit: cover; display: block; background: #0b111c;"
                />
                @if (selectedIndices().has(cand.frameIndex)) {
                  <span
                    style="position: absolute; top: 6px; right: 6px; width: 22px; height: 22px; border-radius: 999px; background: var(--rv-accent); color: #fff; display: grid; place-items: center;"
                  >
                    <mat-icon style="width: 14px; height: 14px; font-size: 14px; line-height: 14px;">check</mat-icon>
                  </span>
                }
                <div style="padding: 6px 8px; font-size: 0.76rem; display: flex; justify-content: space-between; align-items: center;">
                  <span class="rv-mono">#{{ cand.frameIndex }} · {{ (cand.timestampMs / 1000).toFixed(1) }}s</span>
                  <span class="rv-badge rv-badge--neutral">Δ {{ cand.diffScore }}</span>
                </div>
              </div>
            }
          </div>
        }
      }
    </section>
  `,
})
export class AnnotationVideoExtractorComponent {
  private readonly api = inject(AnnotationsApiService);
  private readonly notifications = inject(NotificationService);

  readonly projectId = input.required<string>();
  readonly videos = input<AnnotationVideo[]>([]);
  readonly framesImported = output<void>();

  protected readonly uploadingVideo = signal<boolean>(false);
  protected readonly previewing = signal<boolean>(false);
  protected readonly extracting = signal<boolean>(false);
  protected readonly activeVideoId = signal<string | null>(null);
  protected readonly sampleFps = signal<number>(1);
  protected readonly maxFrames = signal<number>(24);
  protected readonly dedupThreshold = signal<number>(4.0);
  protected readonly candidates = signal<PreviewFrameCandidate[]>([]);
  protected readonly skippedSimilar = signal<number>(0);
  protected readonly selectedIndices = signal<Set<number>>(new Set());

  protected readonly selectedCount = computed(() => this.selectedIndices().size);

  constructor() {
    effect(() => {
      const vids = this.videos();
      if (vids.length > 0 && !this.activeVideoId()) {
        this.activeVideoId.set(vids[0].id);
      }
    });
  }

  protected onVideoUpload(event: Event): void {
    const inputEl = event.target as HTMLInputElement;
    const file = inputEl.files?.[0];
    inputEl.value = '';
    if (!file) {
      return;
    }
    this.uploadingVideo.set(true);
    this.api.uploadVideo(this.projectId(), file).subscribe({
      next: (vid) => {
        this.uploadingVideo.set(false);
        this.activeVideoId.set(vid.id);
        this.notifications.success(`Vídeo '${vid.filename}' pronto para extração de frames.`);
        this.framesImported.emit();
      },
      error: (err) => {
        this.uploadingVideo.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao importar vídeo.'));
      },
    });
  }

  protected generatePreview(): void {
    const vidId = this.activeVideoId();
    if (!vidId) {
      return;
    }
    this.previewing.set(true);
    this.candidates.set([]);
    this.api
      .previewVideoFrames(this.projectId(), vidId, this.sampleFps(), this.maxFrames(), this.dedupThreshold())
      .subscribe({
        next: (res) => {
          this.previewing.set(false);
          this.candidates.set(res.candidates);
          this.skippedSimilar.set(res.skippedSimilar);
          this.selectedIndices.set(new Set(res.candidates.map((c) => c.frameIndex)));
        },
        error: (err) => {
          this.previewing.set(false);
          this.notifications.error(extractApiErrorMessage(err, 'Falha ao gerar miniaturas.'));
        },
      });
  }

  protected toggleCandidate(frameIndex: number): void {
    this.selectedIndices.update((curr) => {
      const next = new Set(curr);
      if (next.has(frameIndex)) {
        next.delete(frameIndex);
      } else {
        next.add(frameIndex);
      }
      return next;
    });
  }

  protected selectAll(checked: boolean): void {
    if (checked) {
      this.selectedIndices.set(new Set(this.candidates().map((c) => c.frameIndex)));
    } else {
      this.selectedIndices.set(new Set());
    }
  }

  protected commitSelected(): void {
    const vidId = this.activeVideoId();
    const indices = Array.from(this.selectedIndices()).sort((a, b) => a - b);
    if (!vidId || indices.length === 0) {
      return;
    }
    this.extracting.set(true);
    this.api.extractVideoFrames(this.projectId(), vidId, { frameIndices: indices }).subscribe({
      next: (created) => {
        this.extracting.set(false);
        this.candidates.set([]);
        this.selectedIndices.set(new Set());
        this.notifications.success(`${created.length} frames extraídos em resolução original.`);
        this.framesImported.emit();
      },
      error: (err) => {
        this.extracting.set(false);
        this.notifications.error(extractApiErrorMessage(err, 'Falha ao extrair frames selecionados.'));
      },
    });
  }
}
