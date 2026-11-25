import { DecimalPipe } from '@angular/common';
import { Component, ElementRef, ViewChild, computed, input, output, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { RouterLink } from '@angular/router';
import {
  CurateFrameRequest,
  CurationExport,
  ImageRunResult,
  LabRunResult,
  VideoFrameSummary,
  VideoRunStatus,
} from '../models/lab.models';

export interface TrackSummaryRow {
  trackId: number;
  observationCount: number;
  firstTimestampMs: number;
  lastTimestampMs: number;
  meanConfidence: number;
}

@Component({
  selector: 'rv-lab-run-viewer',
  standalone: true,
  imports: [
    DecimalPipe,
    FormsModule,
    RouterLink,
    MatIconModule,
  ],
  template: `
    @if (activeRun(); as run) {
      @if (run.status === 'FAILED' && run.mediaType === 'VIDEO') {
        <div class="rv-banner rv-banner--error" role="alert">
          <mat-icon aria-hidden="true">error</mat-icon>
          <div>{{ run.errorMessage || 'Falha ao processar vídeo no laboratório.' }}</div>
        </div>
      } @else if (run.status === 'COMPLETED') {
        <section class="rv-card" data-testid="onnx-run-results">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Resultado — {{ run.originalFilename }}</h3>
              <p class="rv-card__subtitle">
                Modelo <span class="rv-mono">{{ run.modelConfig.modelName }}</span>
                (SHA <span class="rv-mono">{{ run.modelConfig.modelSha256.slice(0, 12) }}</span>) ·
                Conf ≥ {{ run.modelConfig.confThreshold | number: '1.2-2' }} ·
                IoU {{ run.modelConfig.iouThreshold | number: '1.2-2' }}
                @if (run.modelConfig.sampleFps != null) {
                  · {{ run.modelConfig.sampleFps }} fps
                }
              </p>
            </div>
            <div class="rv-header-actions">
              @if (run.files.annotatedDownloadUrl) {
                <a class="rv-btn rv-btn--primary" [href]="run.files.annotatedDownloadUrl" download>
                  <mat-icon>download</mat-icon>
                  <span>{{ run.mediaType === 'IMAGE' ? 'Salvar imagem anotada' : 'Salvar vídeo anotado' }}</span>
                </a>
              }
              <a class="rv-btn rv-btn--secondary" [href]="run.files.originalDownloadUrl" download>
                <mat-icon>insert_drive_file</mat-icon>
                <span>Original</span>
              </a>
              <button type="button" class="rv-btn rv-btn--secondary" (click)="exportRunJson()">
                <mat-icon>code</mat-icon>
                <span>JSON</span>
              </button>
              <button type="button" class="rv-btn rv-btn--secondary" (click)="exportRunCsv()">
                <mat-icon>table_chart</mat-icon>
                <span>CSV</span>
              </button>
            </div>
          </div>

          @if (run.metrics; as m) {
            <div class="rv-kpi-grid" style="margin-bottom: 18px;">
              <div class="rv-kpi-card">
                <div class="rv-kpi-card__body">
                  <span class="rv-kpi-card__label">FPS efetivo</span>
                  <span class="rv-kpi-card__value" data-testid="kpi-fps">{{ m.processingFps | number: '1.1-2' }}</span>
                  <span class="rv-kpi-card__meta">Inferência pura: {{ m.inferenceOnlyFps | number: '1.1-2' }} FPS</span>
                </div>
              </div>
              <div class="rv-kpi-card">
                <div class="rv-kpi-card__body">
                  <span class="rv-kpi-card__label">Latência média / frame</span>
                  <span class="rv-kpi-card__value" style="font-size: 1.5rem;">{{ m.avgFrameMs | number: '1.1-2' }} ms</span>
                  <span class="rv-kpi-card__meta">p50 {{ m.p50FrameMs | number: '1.1-1' }} · p95 {{ m.p95FrameMs | number: '1.1-1' }} ms</span>
                </div>
              </div>
              <div class="rv-kpi-card">
                <div class="rv-kpi-card__body">
                  <span class="rv-kpi-card__label">Detecções encontradas</span>
                  <span class="rv-kpi-card__value" data-testid="kpi-detections">
                    {{ run.mediaType === 'IMAGE' ? run.detectionCount : run.totalDetections }}
                  </span>
                  <span class="rv-kpi-card__meta">
                    @if (run.mediaType === 'VIDEO') {
                      Em {{ run.framesWithDetections }} de {{ run.processedFrames }} frames
                    } @else {
                      Resolução {{ run.imageWidth }}×{{ run.imageHeight }}
                    }
                  </span>
                </div>
              </div>
              <div class="rv-kpi-card">
                <div class="rv-kpi-card__body">
                  <span class="rv-kpi-card__label">Dispositivo utilizado</span>
                  <span class="rv-kpi-card__value" style="font-size: 1.35rem;">{{ run.modelConfig.deviceType }}</span>
                  <span class="rv-kpi-card__meta rv-mono">{{ run.modelConfig.activeExecutionProvider }}</span>
                </div>
              </div>
              <div class="rv-kpi-card">
                <div class="rv-kpi-card__body">
                  <span class="rv-kpi-card__label">Tempo total</span>
                  <span class="rv-kpi-card__value" style="font-size: 1.35rem;">{{ formatDuration(m.totalMs) }}</span>
                  <span class="rv-kpi-card__meta">
                    Pré {{ m.preprocessMs | number: '1.1-1' }} · Inf {{ m.inferenceMs | number: '1.1-1' }} ms
                  </span>
                </div>
              </div>
            </div>
          }

          @if (run.noDetections && run.emptyMessage) {
            <div class="rv-banner rv-banner--warning" role="status" style="margin-bottom: 16px;" data-testid="no-detections-banner">
              <mat-icon aria-hidden="true">search_off</mat-icon>
              <div>
                <strong>Nenhum objeto detectado.</strong>
                {{ run.emptyMessage }}
              </div>
            </div>
          }

          @if (imageRun(); as imgRun) {
            <div class="rv-filter-bar" style="margin-bottom: 14px;">
              <button type="button" class="rv-btn rv-btn--secondary" (click)="toggleOriginalImage.emit()">
                <mat-icon>compare</mat-icon>
                <span>{{ showOriginalImage() ? 'Ocultar original' : 'Comparar com original' }}</span>
              </button>
              <select
                class="rv-select"
                style="width: 220px;"
                aria-label="Motivo da curadoria"
                [ngModel]="curationReason()"
                (ngModelChange)="curationReason.set($event)"
                data-testid="image-curation-reason"
              >
                <option value="false_negative">Falso negativo</option>
                <option value="false_positive">Falso positivo</option>
                <option value="bad_bbox">Bounding box imprecisa</option>
              </select>
              <button
                type="button"
                class="rv-btn rv-btn--secondary"
                data-testid="curate-image-btn"
                [disabled]="curating()"
                (click)="triggerCuration(run.runId, 0)"
              >
                <mat-icon>add_photo_alternate</mat-icon>
                <span>{{ curating() ? 'Exportando...' : 'Enviar frame para curadoria' }}</span>
              </button>
            </div>

            <div
              style="display: grid; gap: 16px; margin-bottom: 16px;"
              [style.grid-template-columns]="showOriginalImage() ? 'repeat(auto-fit, minmax(280px, 1fr))' : '1fr'"
            >
              <div>
                <h4 style="margin: 0 0 8px; font-size: 0.88rem; color: var(--rv-text-secondary);">
                  Imagem anotada (detecção por frame)
                </h4>
                <img
                  [src]="imgRun.files.annotatedUrl!"
                  alt="Imagem com detecções sobrepostas"
                  style="width: 100%; max-height: 520px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                />
              </div>
              @if (showOriginalImage()) {
                <div>
                  <h4 style="margin: 0 0 8px; font-size: 0.88rem; color: var(--rv-text-secondary);">Imagem original</h4>
                  <img
                    [src]="imgRun.files.originalUrl"
                    alt="Imagem original sem alterações"
                    style="width: 100%; max-height: 520px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                  />
                </div>
              }
            </div>

            @if (imgRun.detections.length > 0) {
              <div class="rv-card rv-card--table" style="box-shadow: none;">
                <div class="rv-table-wrap">
                  <table class="rv-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Classe</th>
                        <th>Confiança</th>
                        <th>Tracking</th>
                        <th>Caixa delimitadora (px)</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (det of imgRun.detections; track $index) {
                        <tr>
                          <td>{{ $index + 1 }}</td>
                          <td class="rv-mono">{{ det.className }}</td>
                          <td>{{ det.confidence * 100 | number: '1.1-1' }}%</td>
                          <td>
                            <span class="rv-badge rv-badge--neutral">Não aplicável (imagem única)</span>
                          </td>
                          <td class="rv-mono">
                            [{{ det.x1 | number: '1.0-0' }}, {{ det.y1 | number: '1.0-0' }}, {{ det.x2 | number: '1.0-0' }}, {{ det.y2 | number: '1.0-0' }}]
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            }
          }

          @if (videoRun(); as vidRun) {
            @if (vidRun.truncatedByFrameLimit) {
              <div class="rv-banner rv-banner--warning" style="margin-bottom: 14px;">
                <mat-icon aria-hidden="true">info</mat-icon>
                <div>
                  O vídeo atingiu o limite configurável da interface ({{ vidRun.processedFrames }} frames amostrados).
                </div>
              </div>
            }

            <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 14px;">
              <span class="rv-badge rv-badge--neutral">
                Entrada: {{ vidRun.videoWidth }}×{{ vidRun.videoHeight }} · {{ vidRun.sourceCodec }} · {{ vidRun.sourceFps }} fps
              </span>
              @if (vidRun.outputCodec) {
                <span class="rv-badge rv-badge--info" data-testid="video-output-badge">
                  Saída: {{ vidRun.outputContainer }} / {{ vidRun.outputEncoder ?? vidRun.outputCodec }}
                  @if (vidRun.outputWidth != null && vidRun.outputHeight != null) {
                    · {{ vidRun.outputWidth }}×{{ vidRun.outputHeight }}
                  }
                  @if (vidRun.outputCrf != null) {
                    · CRF {{ vidRun.outputCrf }}
                  }
                </span>
              }
              @if (vidRun.modelConfig.trackingIntegrated) {
                <span class="rv-badge rv-badge--success" data-testid="video-tracking-badge">
                  <span class="rv-badge__dot" aria-hidden="true"></span>
                  Tracking: ByteTrack ativo · {{ vidRun.uniqueTrackCount ?? 0 }} tracks únicos
                </span>
              } @else {
                <span class="rv-badge rv-badge--warning" data-testid="video-tracking-badge">
                  <span class="rv-badge__dot" aria-hidden="true"></span>
                  Tracking desativado (somente detecção por frame)
                </span>
              }
            </div>

            @if (vidRun.files.annotatedUrl) {
              <div class="rv-video-stage" style="margin-bottom: 18px;">
                <video #annotatedVideo controls preload="metadata" [src]="vidRun.files.annotatedUrl">
                  Seu navegador não conseguiu reproduzir o vídeo anotado; utilize "Salvar vídeo anotado".
                </video>
              </div>
            }

            @if (trackSummary().length > 0) {
              <div class="rv-card rv-card--table" style="box-shadow: none; margin-bottom: 18px;">
                <div class="rv-card__header">
                  <h4 class="rv-card__title" style="font-size: 0.92rem;">
                    Tracks associados pelo ByteTrack ({{ trackSummary().length }})
                  </h4>
                </div>
                <div class="rv-table-wrap">
                  <table class="rv-table">
                    <thead>
                      <tr>
                        <th>Track ID</th>
                        <th>Observações</th>
                        <th>Janela temporal</th>
                        <th>Confiança média</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (tr of trackSummary(); track tr.trackId) {
                        <tr class="rv-row--selectable" (click)="seekVideo(tr.firstTimestampMs)">
                          <td class="rv-mono">#{{ tr.trackId }}</td>
                          <td>{{ tr.observationCount }}</td>
                          <td class="rv-mono">{{ tr.firstTimestampMs }} ms → {{ tr.lastTimestampMs }} ms</td>
                          <td>{{ tr.meanConfidence * 100 | number: '1.1-1' }}%</td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            }

            @if (vidRun.frameSummaries.length > 0) {
              @if (vidRun.frameSummariesTruncated) {
                <div class="rv-banner rv-banner--info" style="margin-bottom: 12px;" data-testid="frames-truncated-banner">
                  <mat-icon aria-hidden="true">info</mat-icon>
                  <div>
                    Exibindo os primeiros {{ vidRun.frameSummaries.length }} de
                    {{ vidRun.totalFrameSummaries ?? vidRun.frameSummaries.length }} frames amostrados.
                  </div>
                </div>
              }

              <div class="rv-card rv-card--table" style="box-shadow: none;">
                <div class="rv-card__header">
                  <h4 class="rv-card__title" style="font-size: 0.92rem;">
                    Detecções ao longo dos frames ({{ filteredFrames().length }})
                  </h4>
                  <div class="rv-filter-bar">
                    <select
                      class="rv-select"
                      style="width: 215px;"
                      aria-label="Motivo da curadoria"
                      [ngModel]="curationReason()"
                      (ngModelChange)="curationReason.set($event)"
                    >
                      <option value="false_negative">Falso negativo</option>
                      <option value="false_positive">Falso positivo</option>
                      <option value="bad_bbox">Bounding box imprecisa</option>
                      <option value="id_switch">Troca de ID (tracking)</option>
                    </select>
                    <button type="button" class="rv-btn rv-btn--secondary" (click)="toggleOnlyDetected.emit()">
                      <mat-icon>filter_list</mat-icon>
                      <span>{{ onlyDetectedFrames() ? 'Todos os frames' : 'Apenas com detecções' }}</span>
                    </button>
                  </div>
                </div>

                <div class="rv-table-wrap" style="max-height: 340px; overflow-y: auto;">
                  <table class="rv-table">
                    <thead>
                      <tr>
                        <th>Frame</th>
                        <th>Timestamp</th>
                        <th>Detecções</th>
                        <th>Latência (ms)</th>
                        <th>Maior confiança</th>
                        <th>Curadoria</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (fr of filteredFrames(); track fr.frameIndex) {
                        <tr class="rv-row--selectable" (click)="seekVideo(fr.timestampMs)">
                          <td class="rv-mono">{{ fr.frameIndex }}</td>
                          <td class="rv-mono">{{ fr.timestampMs }} ms</td>
                          <td>{{ fr.detectionCount }}</td>
                          <td>{{ fr.frameLatencyMs | number: '1.1-2' }} ms</td>
                          <td>
                            @if (fr.detections.length > 0) {
                              <span class="rv-mono">
                                {{ fr.detections[0].trackId != null ? '#' + fr.detections[0].trackId + ' · ' : '' }}{{ fr.detections[0].className }} ({{ fr.detections[0].confidence * 100 | number: '1.1-1' }}%)
                              </span>
                            } @else {
                              <span style="color: var(--rv-text-muted);">—</span>
                            }
                          </td>
                          <td>
                            <button
                              type="button"
                              class="rv-btn rv-btn--secondary rv-btn--sm"
                              data-testid="curate-frame-btn"
                              [disabled]="curating()"
                              (click)="$event.stopPropagation(); triggerCuration(vidRun.runId, fr.frameIndex)"
                            >
                              <mat-icon>playlist_add_check</mat-icon>
                              <span>Curar</span>
                            </button>
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              </div>
            }
          }

          @if (lastCuration(); as cur) {
            <div
              class="rv-banner"
              [class.rv-banner--warning]="cur.reservedBenchmarkWarning"
              [class.rv-banner--info]="!cur.reservedBenchmarkWarning"
              style="margin-top: 16px; justify-content: space-between; align-items: center; flex-wrap: wrap;"
              data-testid="curation-result-banner"
            >
              <div style="display: flex; align-items: center; gap: 10px;">
                <mat-icon aria-hidden="true">{{ cur.reservedBenchmarkWarning ? 'warning' : 'check_circle' }}</mat-icon>
                <div>
                  Frame #{{ cur.frameIndex }} enviado para curadoria em
                  <span class="rv-mono">{{ cur.imagePath }}</span>
                  ({{ cur.detectionCount }} pré-anotações · revisão humana pendente).
                  @if (cur.reservedBenchmarkWarning) {
                    <strong> Aviso: mídia pertence ao conjunto reservado (vid_01/vid_09/vid_16); não incorporar a treino/val.</strong>
                  }
                </div>
              </div>
              <a class="rv-btn rv-btn--secondary rv-btn--sm" routerLink="/annotations">Abrir em Anotações</a>
            </div>
          }
        </section>
      }
    }
  `,
})
export class LabRunViewerComponent {
  @ViewChild('annotatedVideo') private videoEl?: ElementRef<HTMLVideoElement>;

  readonly activeRun = input<LabRunResult | null>(null);
  readonly showOriginalImage = input<boolean>(false);
  readonly onlyDetectedFrames = input<boolean>(false);
  readonly curating = input<boolean>(false);
  readonly lastCuration = input<CurationExport | null>(null);

  readonly toggleOriginalImage = output<void>();
  readonly toggleOnlyDetected = output<void>();
  readonly curateRequested = output<{ runId: string; payload: CurateFrameRequest }>();

  readonly curationReason = signal<string>('false_negative');

  readonly imageRun = computed<ImageRunResult | null>(() => {
    const run = this.activeRun();
    return run && run.mediaType === 'IMAGE' ? run : null;
  });

  readonly videoRun = computed<VideoRunStatus | null>(() => {
    const run = this.activeRun();
    return run && run.mediaType === 'VIDEO' ? run : null;
  });

  readonly filteredFrames = computed<VideoFrameSummary[]>(() => {
    const vid = this.videoRun();
    if (!vid) {
      return [];
    }
    return this.onlyDetectedFrames()
      ? vid.frameSummaries.filter((f) => f.detectionCount > 0)
      : vid.frameSummaries;
  });

  readonly trackSummary = computed<TrackSummaryRow[]>(() => {
    const vid = this.videoRun();
    if (!vid || !vid.modelConfig.trackingIntegrated) {
      return [];
    }
    const grouped = new Map<number, { confs: number[]; firstMs: number; lastMs: number }>();
    for (const fr of vid.frameSummaries) {
      for (const det of fr.detections) {
        if (det.trackId == null) {
          continue;
        }
        const entry = grouped.get(det.trackId);
        if (!entry) {
          grouped.set(det.trackId, {
            confs: [det.confidence],
            firstMs: fr.timestampMs,
            lastMs: fr.timestampMs,
          });
        } else {
          entry.confs.push(det.confidence);
          entry.firstMs = Math.min(entry.firstMs, fr.timestampMs);
          entry.lastMs = Math.max(entry.lastMs, fr.timestampMs);
        }
      }
    }
    return Array.from(grouped.entries())
      .sort(([a], [b]) => a - b)
      .map(([trackId, info]) => ({
        trackId,
        observationCount: info.confs.length,
        firstTimestampMs: info.firstMs,
        lastTimestampMs: info.lastMs,
        meanConfidence: info.confs.reduce((acc, v) => acc + v, 0) / info.confs.length,
      }));
  });

  seekVideo(timestampMs: number): void {
    const el = this.videoEl?.nativeElement;
    if (el) {
      el.currentTime = Math.max(0, timestampMs / 1000);
    }
  }

  triggerCuration(runId: string, frameIndex: number): void {
    this.curateRequested.emit({
      runId,
      payload: { frameIndex, reason: this.curationReason() },
    });
  }

  exportRunJson(): void {
    const run = this.activeRun();
    if (!run) {
      return;
    }
    this.triggerDownload(
      new Blob([JSON.stringify(run, null, 2)], { type: 'application/json' }),
      `vision-lab-${run.runId.slice(0, 8)}.json`
    );
  }

  exportRunCsv(): void {
    const run = this.activeRun();
    if (!run) {
      return;
    }
    const lines: string[] = ['frame_index,timestamp_ms,class_name,confidence,track_id,x1,y1,x2,y2'];
    if (run.mediaType === 'IMAGE') {
      for (const d of run.detections) {
        lines.push(`0,0,${d.className},${d.confidence},${d.trackId ?? ''},${d.x1},${d.y1},${d.x2},${d.y2}`);
      }
    } else {
      for (const fr of run.frameSummaries) {
        for (const d of fr.detections) {
          lines.push(
            `${fr.frameIndex},${fr.timestampMs},${d.className},${d.confidence},${d.trackId ?? ''},${d.x1},${d.y1},${d.x2},${d.y2}`
          );
        }
      }
    }
    this.triggerDownload(
      new Blob([lines.join('\r\n')], { type: 'text/csv;charset=utf-8' }),
      `vision-lab-${run.runId.slice(0, 8)}.csv`
    );
  }

  formatDuration(totalMs: number): string {
    return totalMs >= 1000 ? `${(totalMs / 1000).toFixed(2)} s` : `${totalMs.toFixed(1)} ms`;
  }

  private triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}
