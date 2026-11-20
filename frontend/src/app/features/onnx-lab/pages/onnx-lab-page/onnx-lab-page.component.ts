import { DatePipe, DecimalPipe } from '@angular/common';
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSelectModule } from '@angular/material/select';
import { Subscription, switchMap, takeWhile, timer } from 'rxjs';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';
import {
  ImageRunResult,
  LabRunResult,
  RunListItem,
  RuntimeStatus,
  VideoRunStatus,
} from '../../models/onnx-lab.models';
import { OnnxLabApiService } from '../../services/onnx-lab-api.service';

const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'bmp', 'webp']);
const VIDEO_EXTS = new Set(['mp4', 'webm', 'ogv', 'mov', 'mkv', 'avi']);

@Component({
  selector: 'rv-onnx-lab-page',
  standalone: true,
  imports: [
    FormsModule,
    DatePipe,
    DecimalPipe,
    MatButtonModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatSelectModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    EmptyStateComponent,
    FormatBytesPipe,
  ],
  template: `
    <div class="rv-page">
      <section class="rv-banner rv-banner--warning" role="note" data-testid="onnx-lab-scope-banner">
        <mat-icon aria-hidden="true">science</mat-icon>
        <div style="display: flex; flex-direction: column; gap: 4px;">
          <strong>Laboratório Local de Visão Computacional (Python + FastAPI)</strong>
          <span>
            Inferência local executada via PyTorch CUDA (padrão), PyTorch CPU ou ONNX Runtime CPU com decodificação e codificação controlada PyAV.
            Os arquivos originais selecionados são preservados intactos; datasets e anotações não são alterados.
          </span>
          @if (status(); as st) {
            <span><strong>Limitação do modelo:</strong> {{ st.limitationsNotice }}</span>
            <span data-testid="tracking-limitation-notice">
              <strong>Tracking:</strong> {{ st.trackingNotice }}
            </span>
          }
        </div>
      </section>

      @if (errorMessage()) {
        <div class="rv-banner rv-banner--error" role="alert">
          <mat-icon aria-hidden="true">error_outline</mat-icon>
          <div>{{ errorMessage() }}</div>
        </div>
      }

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Runtime e Modelo Ativo</h2>
            <p class="rv-card__subtitle">
              Dispositivo verificado explicitamente em tempo de execução (sem presunção silenciosa de CUDA).
            </p>
          </div>
          @if (status(); as st) {
            <span class="rv-badge" [class.rv-badge--success]="st.available" [class.rv-badge--error]="!st.available">
              <mat-icon style="font-size: 16px; width: 16px; height: 16px;">memory</mat-icon>
              {{ st.available ? st.deviceType + ' (' + st.activeExecutionProvider + ')' : 'Indisponível' }}
            </span>
          }
        </div>

        @if (loadingStatus()) {
          <mat-progress-bar mode="indeterminate" />
        } @else if (status(); as st) {
          @if (!st.available) {
            <div class="rv-banner rv-banner--error" role="alert">
              <mat-icon aria-hidden="true">warning</mat-icon>
              <div>{{ st.unavailableReason }}</div>
            </div>
          } @else {
            <div class="rv-kpi-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));">
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Modelo ativo</span>
                <span class="rv-mono" style="font-weight: 600;">{{ st.modelName }}</span>
                <span style="font-size: 0.78rem; color: var(--rv-text-muted);">SHA {{ st.modelSha256.slice(0, 12) }}</span>
              </div>
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Dispositivo / Provedor</span>
                <span style="font-weight: 700;">{{ st.deviceType }}</span>
                <span class="rv-mono" style="font-size: 0.78rem; color: var(--rv-text-muted);">{{ st.activeExecutionProvider }}</span>
              </div>
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Tensor de Entrada</span>
                <span class="rv-mono" style="font-weight: 600;">{{ st.inputShape.join(' × ') }}</span>
                <span style="font-size: 0.78rem; color: var(--rv-text-muted);">Saída {{ st.outputShape.join(' × ') }}</span>
              </div>
              <div class="rv-kpi-card">
                <span class="rv-kpi-card__label">Rastreamento (Tracker)</span>
                @if (st.trackingAlgorithms?.includes('bytetrack')) {
                  <span class="rv-badge rv-badge--info" style="align-self: flex-start;">ByteTrack opcional</span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">Habilitável para vídeo</span>
                } @else {
                  <span class="rv-badge rv-badge--warning" style="align-self: flex-start;">Não integrado</span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">Somente detecção por frame</span>
                }
              </div>
            </div>
            <p style="margin: 12px 0 0; font-size: 0.84rem; color: var(--rv-text-secondary);">
              {{ st.deviceDiagnostic }}
            </p>
          }
        }
      </section>

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Testar Imagem ou Vídeo Local</h2>
            <p class="rv-card__subtitle">
              Selecione qualquer imagem (JPG, PNG, BMP, WEBP) ou vídeo (MP4, WEBM, OGV, MOV, MKV, AVI) do computador.
              Alterações aqui são apenas exploratórias e não servem para selecionar hiperparâmetros sobre o conjunto de teste.
            </p>
          </div>
        </div>

        <form (ngSubmit)="runAnalysis()" style="display: flex; flex-direction: column; gap: 14px;">
          <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px;">
            <label
              style="display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; border: 1px dashed var(--rv-border-strong); border-radius: var(--rv-radius-sm); cursor: pointer; background: var(--rv-bg-surface-muted);"
            >
              <mat-icon aria-hidden="true">{{ selectedMediaKind() === 'VIDEO' ? 'video_file' : 'image' }}</mat-icon>
              <span>{{ selectedFile() ? selectedFile()!.name : 'Selecionar imagem ou vídeo local' }}</span>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.webm,.ogv,.mov,.mkv,.avi,image/*,video/*"
                (change)="onFileSelected($event)"
                style="display: none;"
              />
            </label>

            @if (selectedFile(); as f) {
              <span class="rv-badge rv-badge--neutral">
                {{ selectedMediaKind() === 'VIDEO' ? 'Vídeo' : selectedMediaKind() === 'IMAGE' ? 'Imagem' : 'Arquivo' }}
              </span>
              <span style="font-size: 0.85rem; color: var(--rv-text-secondary);">
                {{ f.size | formatBytes }}
              </span>
            }
          </div>

          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px;">
            <mat-form-field appearance="outline">
              <mat-label>Confiança mínima (exploratório)</mat-label>
              <input
                matInput
                type="number"
                min="0.05"
                max="0.95"
                step="0.05"
                name="confThreshold"
                [ngModel]="confThreshold()"
                (ngModelChange)="confThreshold.set(toBoundedNumber($event, 0.25, 0.05, 0.95))"
              />
            </mat-form-field>

            <mat-form-field appearance="outline">
              <mat-label>IoU NMS (exploratório)</mat-label>
              <input
                matInput
                type="number"
                min="0.10"
                max="0.90"
                step="0.05"
                name="iouThreshold"
                [ngModel]="iouThreshold()"
                (ngModelChange)="iouThreshold.set(toBoundedNumber($event, 0.45, 0.10, 0.90))"
              />
            </mat-form-field>

            @if (status()?.availableRuntimes; as runtimeOpts) {
              <mat-form-field appearance="outline">
                <mat-label>Runtime de inferência</mat-label>
                <mat-select
                  name="selectedRuntime"
                  data-testid="runtime-select"
                  [ngModel]="selectedRuntime()"
                  (ngModelChange)="selectedRuntime.set($event)"
                >
                  @for (opt of runtimeOpts; track opt.id) {
                    <mat-option [value]="opt.id" [disabled]="!opt.available">
                      {{ opt.label }}
                    </mat-option>
                  }
                </mat-select>
              </mat-form-field>
            }

            @if (selectedMediaKind() === 'VIDEO') {
              <mat-form-field appearance="outline">
                <mat-label>Amostragem do vídeo</mat-label>
                <mat-select
                  name="sampleFps"
                  [ngModel]="sampleFps()"
                  (ngModelChange)="sampleFps.set($event)"
                >
                  <mat-option [value]="5">5 fps (Recomendado — padrão PoC)</mat-option>
                  <mat-option [value]="2">2 fps (Pré-visualização rápida)</mat-option>
                  <mat-option [value]="10">10 fps (Maior detalhe)</mat-option>
                  <mat-option [value]="0">Todos os frames (FPS nativo)</mat-option>
                </mat-select>
              </mat-form-field>

              @if (status()?.availableOutputResolutions?.length) {
                <mat-form-field appearance="outline">
                  <mat-label>Resolução do vídeo anotado</mat-label>
                  <mat-select
                    name="outputResolution"
                    data-testid="output-resolution-select"
                    [ngModel]="outputResolution()"
                    (ngModelChange)="outputResolution.set($event)"
                  >
                    <mat-option value="1080p">1080p (Padrão H.264/MP4)</mat-option>
                    <mat-option value="native">Resolução nativa (ex.: 4K)</mat-option>
                  </mat-select>
                </mat-form-field>
              }

              @if (status()?.trackingAlgorithms?.includes('bytetrack')) {
                <mat-form-field appearance="outline">
                  <mat-label>Tracking de vagões</mat-label>
                  <mat-select
                    name="enableTracking"
                    data-testid="tracking-select"
                    [ngModel]="enableTracking()"
                    (ngModelChange)="enableTracking.set($event)"
                  >
                    <mat-option [value]="false">Desativado (somente detecção)</mat-option>
                    <mat-option [value]="true">ByteTrack (IDs persistentes)</mat-option>
                  </mat-select>
                </mat-form-field>
              }
            }
          </div>

          <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
            <button
              type="submit"
              mat-flat-button
              color="primary"
              [disabled]="!canSubmit()"
            >
              <mat-icon>play_circle</mat-icon>
              {{ submitting() ? 'Enviando...' : 'Executar inferência' }}
            </button>
          </div>

          @if (submitting()) {
            <mat-progress-bar mode="indeterminate" />
          }
        </form>
      </section>

      @if (activeVideoProgress(); as vp) {
        <section class="rv-card" data-testid="video-progress-card">
          <div class="rv-card__header">
            <div>
              <h2 class="rv-card__title">Processando vídeo em Python + PyAV...</h2>
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

      @if (activeRun(); as run) {
        @if (run.status === 'FAILED' && run.mediaType === 'VIDEO') {
          <div class="rv-banner rv-banner--error" role="alert">
            <mat-icon aria-hidden="true">error</mat-icon>
            <div>{{ run.errorMessage || 'Falha ao processar vídeo no laboratório de visão.' }}</div>
          </div>
        } @else if (run.status === 'COMPLETED') {
          <section class="rv-card" data-testid="onnx-run-results">
            <div class="rv-card__header">
              <div>
                <h2 class="rv-card__title">Resultado — {{ run.originalFilename }}</h2>
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
              <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                @if (run.files.annotatedDownloadUrl) {
                  <a mat-flat-button color="primary" [href]="run.files.annotatedDownloadUrl" download>
                    <mat-icon>download</mat-icon>
                    {{ run.mediaType === 'IMAGE' ? 'Salvar imagem anotada' : 'Salvar vídeo anotado' }}
                  </a>
                }
                <a mat-stroked-button [href]="run.files.originalDownloadUrl" download>
                  <mat-icon>file_present</mat-icon>
                  Baixar original
                </a>
              </div>
            </div>

            @if (run.metrics; as m) {
              <div class="rv-kpi-grid" style="margin-bottom: 16px;">
                <div class="rv-kpi-card">
                  <span class="rv-kpi-card__label">FPS de processamento</span>
                  <span class="rv-kpi-card__value" data-testid="kpi-fps">{{ m.processingFps | number: '1.1-2' }}</span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">Inferência pura: {{ m.inferenceOnlyFps | number: '1.1-2' }} FPS</span>
                </div>
                <div class="rv-kpi-card">
                  <span class="rv-kpi-card__label">Latência média / frame</span>
                  <span class="rv-kpi-card__value">{{ m.avgFrameMs | number: '1.1-2' }} ms</span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">p95: {{ m.p95FrameMs | number: '1.1-2' }} ms</span>
                </div>
                <div class="rv-kpi-card">
                  <span class="rv-kpi-card__label">Detecções encontradas</span>
                  <span class="rv-kpi-card__value" data-testid="kpi-detections">
                    {{ run.mediaType === 'IMAGE' ? run.detectionCount : run.totalDetections }}
                  </span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">
                    @if (run.mediaType === 'VIDEO') {
                      Em {{ run.framesWithDetections }} de {{ run.processedFrames }} frames
                    } @else {
                      {{ run.imageWidth }} × {{ run.imageHeight }} px
                    }
                  </span>
                </div>
                <div class="rv-kpi-card">
                  <span class="rv-kpi-card__label">Dispositivo utilizado</span>
                  <span class="rv-kpi-card__value" style="font-size: 1.35rem;">{{ run.modelConfig.deviceType }}</span>
                  <span class="rv-mono" style="font-size: 0.78rem; color: var(--rv-text-muted);">{{ run.modelConfig.activeExecutionProvider }}</span>
                </div>
                <div class="rv-kpi-card">
                  <span class="rv-kpi-card__label">Tempo total</span>
                  <span class="rv-kpi-card__value">{{ formatDuration(m.totalMs) }}</span>
                  <span style="font-size: 0.78rem; color: var(--rv-text-muted);">Pré {{ m.preprocessMs | number: '1.1-1' }} ms · Inf {{ m.inferenceMs | number: '1.1-1' }} ms</span>
                </div>
              </div>
            }

            @if (run.noDetections) {
              <div class="rv-banner rv-banner--warning" role="status" data-testid="no-detections-banner" style="margin-bottom: 16px;">
                <mat-icon aria-hidden="true">search_off</mat-icon>
                <div>
                  <strong>Nenhum objeto detectado.</strong>
                  {{ run.emptyMessage }}
                </div>
              </div>
            }

            @if (asImageRun(run); as imgRun) {
              <div style="display: flex; gap: 8px; margin-bottom: 12px;">
                <button
                  type="button"
                  mat-stroked-button
                  (click)="showOriginalImage.set(!showOriginalImage())"
                >
                  <mat-icon>compare</mat-icon>
                  {{ showOriginalImage() ? 'Mostrar imagem anotada' : 'Comparar com original' }}
                </button>
              </div>

              <div
                style="display: grid; gap: 16px; margin-bottom: 16px;"
                [style.grid-template-columns]="showOriginalImage() ? 'repeat(auto-fit, minmax(280px, 1fr))' : '1fr'"
              >
                <div>
                  <h3 style="margin: 0 0 8px; font-size: 0.92rem;">Imagem anotada (bounding boxes)</h3>
                  <img
                    [src]="imgRun.files.annotatedUrl!"
                    alt="Imagem com detecções sobrepostas"
                    style="width: 100%; max-height: 540px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                  />
                </div>
                @if (showOriginalImage()) {
                  <div>
                    <h3 style="margin: 0 0 8px; font-size: 0.92rem;">Imagem original (preservada)</h3>
                    <img
                      [src]="imgRun.files.originalUrl"
                      alt="Imagem original sem alterações"
                      style="width: 100%; max-height: 540px; object-fit: contain; border-radius: var(--rv-radius-md); border: 1px solid var(--rv-border); background: #060b13;"
                    />
                  </div>
                }
              </div>

              @if (imgRun.detections.length > 0) {
                <div class="rv-table-wrap">
                  <table class="rv-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Classe</th>
                        <th>Confiança</th>
                        <th>ID do vagão (Tracker)</th>
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
                            <span class="rv-badge rv-badge--neutral">
                              {{ det.trackId != null ? '#' + det.trackId : 'N/D (não integrado)' }}
                            </span>
                          </td>
                          <td class="rv-mono">
                            [{{ det.x1 | number: '1.0-0' }}, {{ det.y1 | number: '1.0-0' }}, {{ det.x2 | number: '1.0-0' }}, {{ det.y2 | number: '1.0-0' }}]
                          </td>
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              }
            }

            @if (asVideoRun(run); as vidRun) {
              @if (vidRun.truncatedByFrameLimit) {
                <div class="rv-banner rv-banner--warning" style="margin-bottom: 12px;">
                  <mat-icon aria-hidden="true">info</mat-icon>
                  <div>
                    O vídeo atingiu o limite configurável da interface ({{ vidRun.processedFrames }} frames amostrados).
                    Use taxa de amostragem menor ou o runner de benchmark para execuções longas.
                  </div>
                </div>
              }

              <div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 10px;">
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
                    ByteTrack ativo · {{ vidRun.uniqueTrackCount ?? 0 }} tracks únicos
                  </span>
                } @else {
                  <span class="rv-badge rv-badge--warning" data-testid="video-tracking-badge">
                    IDs de vagão: tracking desativado nesta execução
                  </span>
                }
              </div>

              @if (vidRun.files.annotatedUrl) {
                <div class="rv-video-stage" style="margin-bottom: 16px;">
                  <video controls preload="metadata" [src]="vidRun.files.annotatedUrl">
                    Seu navegador não conseguiu reproduzir o vídeo anotado inline; utilize "Salvar vídeo anotado".
                  </video>
                </div>
              }

              @if (vidRun.frameSummaries.length > 0) {
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
                  <h3 style="margin: 0; font-size: 0.95rem;">Detecções ao longo dos frames ({{ filteredFrames().length }})</h3>
                  <button
                    type="button"
                    mat-stroked-button
                    (click)="onlyDetectedFrames.set(!onlyDetectedFrames())"
                  >
                    <mat-icon>filter_list</mat-icon>
                    {{ onlyDetectedFrames() ? 'Mostrar todos os frames amostrados' : 'Mostrar apenas frames com detecções' }}
                  </button>
                </div>

                <div class="rv-table-wrap" style="max-height: 320px; overflow-y: auto;">
                  <table class="rv-table">
                    <thead>
                      <tr>
                        <th>Frame</th>
                        <th>Timestamp</th>
                        <th>Detecções</th>
                        <th>Latência (ms)</th>
                        <th>Maior confiança</th>
                      </tr>
                    </thead>
                    <tbody>
                      @for (fr of filteredFrames(); track fr.frameIndex) {
                        <tr>
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
                        </tr>
                      }
                    </tbody>
                  </table>
                </div>
              }
            }
          </section>
        }
      }

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Histórico de Testes Locais no Laboratório</h2>
            <p class="rv-card__subtitle">
              Armazenamento isolado em <span class="rv-mono">data/processed/onnx-spike-runs/</span>.
            </p>
          </div>
          <button type="button" mat-stroked-button (click)="refreshRuns()">
            <mat-icon>refresh</mat-icon>
            Atualizar
          </button>
        </div>

        @if (runs().length === 0) {
          <rv-empty-state
            icon="biotech"
            title="Nenhum teste executado ainda"
            message="Selecione uma imagem ou vídeo acima para executar inferência local em GPU ou CPU."
          />
        } @else {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Arquivo</th>
                  <th>Tipo</th>
                  <th>Status</th>
                  <th>Detecções</th>
                  <th>FPS / Latência</th>
                  <th>Dispositivo</th>
                  <th>Data</th>
                </tr>
              </thead>
              <tbody>
                @for (item of runs(); track item.runId) {
                  <tr
                    class="rv-row--selectable"
                    [class.rv-row--selected]="activeRun()?.runId === item.runId"
                    (click)="selectRun(item.runId)"
                  >
                    <td>{{ item.originalFilename }}</td>
                    <td>{{ item.mediaType === 'IMAGE' ? 'Imagem' : 'Vídeo' }}</td>
                    <td><rv-status-badge [status]="item.status" /></td>
                    <td>{{ item.noDetections ? '0 (nenhum)' : item.totalDetections }}</td>
                    <td>
                      @if (item.processingFps != null) {
                        {{ item.processingFps | number: '1.1-1' }} FPS ({{ item.avgFrameMs | number: '1.1-1' }} ms)
                      } @else {
                        —
                      }
                    </td>
                    <td class="rv-mono">{{ item.deviceType }} ({{ item.activeExecutionProvider }})</td>
                    <td>{{ item.createdAt | date: 'dd/MM HH:mm:ss' }}</td>
                  </tr>
                }
              </tbody>
            </table>
          </div>
        }
      </section>
    </div>
  `,
})
export class OnnxLabPageComponent implements OnInit {
  private readonly api = inject(OnnxLabApiService);
  private readonly notifications = inject(NotificationService);
  private readonly destroyRef = inject(DestroyRef);

  readonly loadingStatus = signal(true);
  readonly status = signal<RuntimeStatus | null>(null);
  readonly selectedFile = signal<File | null>(null);
  readonly confThreshold = signal(0.25);
  readonly iouThreshold = signal(0.45);
  readonly sampleFps = signal<number>(5);
  readonly selectedRuntime = signal<string>('pytorch_cuda');
  readonly outputResolution = signal<string>('1080p');
  readonly enableTracking = signal<boolean>(false);
  readonly submitting = signal(false);
  readonly errorMessage = signal<string | null>(null);
  readonly activeRun = signal<LabRunResult | null>(null);
  readonly runs = signal<RunListItem[]>([]);
  readonly showOriginalImage = signal(false);
  readonly onlyDetectedFrames = signal(false);

  private pollSubscription: Subscription | null = null;

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
    const st = this.status();
    const kind = this.selectedMediaKind();
    return Boolean(st?.available && !this.submitting() && (kind === 'IMAGE' || kind === 'VIDEO'));
  });

  readonly activeVideoProgress = computed<VideoRunStatus | null>(() => {
    const run = this.activeRun();
    if (run && run.mediaType === 'VIDEO' && (run.status === 'QUEUED' || run.status === 'RUNNING')) {
      return run;
    }
    return null;
  });

  readonly filteredFrames = computed(() => {
    const run = this.activeRun();
    if (!run || run.mediaType !== 'VIDEO') {
      return [];
    }
    if (!this.onlyDetectedFrames()) {
      return run.frameSummaries;
    }
    return run.frameSummaries.filter((f) => f.detectionCount > 0);
  });

  constructor() {
    this.destroyRef.onDestroy(() => this.stopPolling());
  }

  ngOnInit(): void {
    this.api
      .getStatus()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (st) => {
          this.status.set(st);
          this.confThreshold.set(st.defaultConfThreshold);
          this.iouThreshold.set(st.defaultIouThreshold);
          if (st.defaultRuntime) {
            this.selectedRuntime.set(st.defaultRuntime);
          }
          if (st.defaultOutputResolution) {
            this.outputResolution.set(st.defaultOutputResolution);
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
    this.refreshRuns();
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files && input.files.length > 0 ? input.files[0] : null;
    this.selectedFile.set(file);
    this.errorMessage.set(null);
  }

  runAnalysis(): void {
    const file = this.selectedFile();
    const kind = this.selectedMediaKind();
    if (!file || (kind !== 'IMAGE' && kind !== 'VIDEO')) {
      this.errorMessage.set('Selecione uma imagem ou vídeo em formato suportado.');
      return;
    }

    this.submitting.set(true);
    this.errorMessage.set(null);
    this.showOriginalImage.set(false);
    this.stopPolling();

    const st = this.status();
    const runtimeArg = st?.availableRuntimes?.length ? this.selectedRuntime() : null;

    if (kind === 'IMAGE') {
      this.api
        .analyzeImage(file, this.confThreshold(), this.iouThreshold(), runtimeArg)
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
            const msg = extractApiErrorMessage(err, 'Falha ao analisar imagem no laboratório local.');
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
        trackingArg
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
          const msg = extractApiErrorMessage(err, 'Falha ao enviar vídeo para análise no laboratório local.');
          this.errorMessage.set(msg);
          this.notifications.error(msg);
        },
      });
  }

  selectRun(runId: string): void {
    this.stopPolling();
    this.errorMessage.set(null);
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
    this.api
      .listRuns()
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (list) => this.runs.set(list),
        error: () => {
          // status card already reports backend availability errors
        },
      });
  }

  asImageRun(run: LabRunResult): ImageRunResult | null {
    return run.mediaType === 'IMAGE' ? run : null;
  }

  asVideoRun(run: LabRunResult): VideoRunStatus | null {
    return run.mediaType === 'VIDEO' ? run : null;
  }

  toBoundedNumber(raw: unknown, fallback: number, min: number, max: number): number {
    const val = Number(raw);
    if (!Number.isFinite(val)) {
      return fallback;
    }
    return Math.max(min, Math.min(max, val));
  }

  formatDuration(totalMs: number): string {
    if (totalMs >= 1000) {
      return `${(totalMs / 1000).toFixed(2)} s`;
    }
    return `${totalMs.toFixed(1)} ms`;
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
            this.notifications.success('Vídeo anotado pronto no Laboratório de Visão.');
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
