import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { FormatBytesPipe } from '../../../shared/pipes/format-bytes.pipe';
import { ModelCheckpoint, RuntimeStatus } from '../models/lab.models';

@Component({
  selector: 'rv-lab-run-form',
  standalone: true,
  imports: [
    FormsModule,
    MatIconModule,
    MatProgressBarModule,
    FormatBytesPipe,
  ],
  template: `
    <section class="rv-card">
      <div class="rv-card__header">
        <div>
          <h3 class="rv-card__title">Testar imagem ou vídeo local</h3>
          <p class="rv-card__subtitle">
            Execuções exploratórias locais em PyTorch CUDA, PyTorch CPU ou ONNX Runtime com preservação dos arquivos originais.
          </p>
        </div>
      </div>

      <form (ngSubmit)="submitRun.emit()" style="display: flex; flex-direction: column; gap: 16px;">
        <div class="rv-dropzone" (click)="mediaInput.click()">
          <div class="rv-dropzone__left">
            <div class="rv-dropzone__icon">
              <mat-icon aria-hidden="true">{{ selectedMediaKind() === 'VIDEO' ? 'movie' : 'image' }}</mat-icon>
            </div>
            <div>
              <div class="rv-dropzone__title">
                {{ selectedFile() ? selectedFile()!.name : 'Selecionar imagem ou vídeo local para inferência' }}
              </div>
              <div class="rv-dropzone__hint">
                @if (selectedFile(); as f) {
                  {{ selectedMediaKind() === 'VIDEO' ? 'Vídeo' : selectedMediaKind() === 'IMAGE' ? 'Imagem' : 'Arquivo' }} ·
                  {{ f.size | formatBytes }}
                } @else {
                  Imagens (JPG, PNG, BMP, WEBP) ou vídeos (MP4, WEBM, OGV, MOV, MKV, AVI)
                }
              </div>
            </div>
          </div>

          <input
            #mediaInput
            type="file"
            accept=".jpg,.jpeg,.png,.bmp,.webp,.mp4,.webm,.ogv,.mov,.mkv,.avi,image/*,video/*"
            (change)="fileSelected.emit($event)"
            style="display: none;"
          />

          <span class="rv-btn rv-btn--secondary rv-btn--sm">
            <mat-icon>folder_open</mat-icon>
            <span>Escolher arquivo</span>
          </span>
        </div>

        <div class="rv-form-grid">
          @if (checkpoints().length > 0) {
            <div class="rv-field">
              <label class="rv-label" for="lab-model-select">Checkpoint / Modelo</label>
              <select
                id="lab-model-select"
                class="rv-select"
                name="selectedModelId"
                data-testid="model-select"
                [ngModel]="selectedModelId()"
                (ngModelChange)="modelChanged.emit($event)"
              >
                @for (cp of checkpoints(); track cp.id) {
                  <option [ngValue]="cp.id">{{ cp.id }} [{{ cp.status }}]</option>
                }
              </select>
            </div>
          }

          <div class="rv-field">
            <label class="rv-label" for="lab-conf-input">Confiança mínima</label>
            <input
              id="lab-conf-input"
              class="rv-input"
              type="number"
              min="0.01"
              max="0.99"
              step="0.05"
              name="confThreshold"
              [ngModel]="confThreshold()"
              (ngModelChange)="confDraftChanged.emit($event)"
              (blur)="thresholdsBlurred.emit()"
            />
          </div>

          <div class="rv-field">
            <label class="rv-label" for="lab-iou-input">IoU NMS</label>
            <input
              id="lab-iou-input"
              class="rv-input"
              type="number"
              min="0.05"
              max="0.95"
              step="0.05"
              name="iouThreshold"
              [ngModel]="iouThreshold()"
              (ngModelChange)="iouDraftChanged.emit($event)"
              (blur)="thresholdsBlurred.emit()"
            />
          </div>

          @if (status()?.availableRuntimes; as runtimeOpts) {
            <div class="rv-field">
              <label class="rv-label" for="lab-runtime-select">Runtime de inferência</label>
              <select
                id="lab-runtime-select"
                class="rv-select"
                name="selectedRuntime"
                data-testid="runtime-select"
                [ngModel]="selectedRuntime()"
                (ngModelChange)="runtimeChanged.emit($event)"
              >
                @for (opt of runtimeOpts; track opt.id) {
                  <option
                    [ngValue]="opt.id"
                    [disabled]="!opt.available && !(opt.id === 'pytorch_cuda' && allowCpuFallback())"
                  >
                    {{ opt.label }}
                  </option>
                }
              </select>
            </div>
          }

          @if (selectedMediaKind() === 'VIDEO') {
            <div class="rv-field">
              <label class="rv-label" for="lab-fps-select">Amostragem do vídeo</label>
              <select
                id="lab-fps-select"
                class="rv-select"
                name="sampleFps"
                [ngModel]="sampleFps()"
                (ngModelChange)="sampleFpsChanged.emit($event)"
              >
                <option [ngValue]="5">5 fps (Recomendado)</option>
                <option [ngValue]="2">2 fps (Pré-visualização rápida)</option>
                <option [ngValue]="10">10 fps (Maior detalhe)</option>
                <option [ngValue]="0">Todos os frames (FPS nativo)</option>
              </select>
            </div>

            @if (status()?.availableOutputResolutions?.length) {
              <div class="rv-field">
                <label class="rv-label" for="lab-outres-select">Resolução do vídeo anotado</label>
                <select
                  id="lab-outres-select"
                  class="rv-select"
                  name="outputResolution"
                  data-testid="output-resolution-select"
                  [ngModel]="outputResolution()"
                  (ngModelChange)="outputResolutionChanged.emit($event)"
                >
                  <option value="1080p">1080p (Padrão H.264/MP4)</option>
                  <option value="native">Resolução nativa (ex.: 4K)</option>
                </select>
              </div>
            }

            @if (status()?.trackingAlgorithms?.includes('bytetrack')) {
              <div class="rv-field">
                <label class="rv-label" for="lab-tracking-select">Tracking de vagões</label>
                <select
                  id="lab-tracking-select"
                  class="rv-select"
                  name="enableTracking"
                  data-testid="tracking-select"
                  [ngModel]="enableTracking()"
                  (ngModelChange)="trackingChanged.emit($event)"
                >
                  <option [ngValue]="false">Desativado (somente detecção)</option>
                  <option [ngValue]="true">ByteTrack (IDs persistentes)</option>
                </select>
              </div>
            }
          }
        </div>

        <div style="display: flex; align-items: center; gap: 18px; flex-wrap: wrap;">
          <button
            type="submit"
            class="rv-btn rv-btn--primary"
            [disabled]="!canSubmit()"
          >
            <mat-icon>play_arrow</mat-icon>
            <span>{{ submitting() ? 'Enviando...' : 'Executar inferência' }}</span>
          </button>

          @if (status() && !status()!.cudaAvailable) {
            <label class="rv-checkbox-label" data-testid="allow-cpu-fallback-checkbox">
              <input
                type="checkbox"
                name="allowCpuFallback"
                [ngModel]="allowCpuFallback()"
                (ngModelChange)="allowCpuFallbackChanged.emit($event)"
              />
              <span>Permitir fallback explícito para PyTorch CPU</span>
            </label>
          }
        </div>

        @if (submitting()) {
          <mat-progress-bar mode="indeterminate" />
        }
      </form>
    </section>
  `,
})
export class LabRunFormComponent {
  readonly status = input<RuntimeStatus | null>(null);
  readonly checkpoints = input<ModelCheckpoint[]>([]);
  readonly selectedModelId = input<string>('exp04c_yolo26n_opentop_640');
  readonly selectedFile = input<File | null>(null);
  readonly selectedMediaKind = input<'IMAGE' | 'VIDEO' | 'UNKNOWN' | null>(null);
  readonly confThreshold = input<number>(0.25);
  readonly iouThreshold = input<number>(0.45);
  readonly selectedRuntime = input<string>('pytorch_cuda');
  readonly sampleFps = input<number>(5);
  readonly outputResolution = input<string>('1080p');
  readonly enableTracking = input<boolean>(false);
  readonly allowCpuFallback = input<boolean>(false);
  readonly submitting = input<boolean>(false);
  readonly canSubmit = input<boolean>(false);

  readonly fileSelected = output<Event>();
  readonly modelChanged = output<string>();
  readonly confDraftChanged = output<unknown>();
  readonly iouDraftChanged = output<unknown>();
  readonly thresholdsBlurred = output<void>();
  readonly runtimeChanged = output<string>();
  readonly sampleFpsChanged = output<number>();
  readonly outputResolutionChanged = output<string>();
  readonly trackingChanged = output<boolean>();
  readonly allowCpuFallbackChanged = output<boolean>();
  readonly submitRun = output<void>();
}
