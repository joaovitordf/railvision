import { DatePipe } from '@angular/common';
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatIconModule } from '@angular/material/icon';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatSelectModule } from '@angular/material/select';
import { Router, RouterLink } from '@angular/router';
import { InspectionStatus, InspectionSummary } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';

@Component({
  selector: 'rv-inspections-page',
  standalone: true,
  imports: [
    FormsModule,
    RouterLink,
    DatePipe,
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
      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Nova inspeção por vídeo</h2>
            <p class="rv-card__subtitle">
              Envie um vídeo pré-gravado (MP4, MOV, MKV, WEBM ou AVI) para criar uma inspeção local.
            </p>
          </div>
        </div>

        <form (ngSubmit)="submitUpload()" style="display: flex; flex-direction: column; gap: 14px;">
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
            <mat-form-field appearance="outline">
              <mat-label>Título da inspeção (opcional)</mat-label>
              <input
                matInput
                name="title"
                [ngModel]="title()"
                (ngModelChange)="title.set($event)"
                maxlength="200"
                placeholder="Ex.: Pátio Norte - Composição 01"
              />
            </mat-form-field>

            <mat-form-field appearance="outline">
              <mat-label>Observações (opcional)</mat-label>
              <input
                matInput
                name="notes"
                [ngModel]="notes()"
                (ngModelChange)="notes.set($event)"
                maxlength="1000"
                placeholder="Contexto de câmera ou iluminação"
              />
            </mat-form-field>
          </div>

          <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 12px;">
            <label
              style="display: inline-flex; align-items: center; gap: 8px; padding: 9px 14px; border: 1px dashed var(--rv-border-strong); border-radius: var(--rv-radius-sm); cursor: pointer; background: var(--rv-bg-surface-muted);"
            >
              <mat-icon aria-hidden="true">video_file</mat-icon>
              <span>{{ selectedFile() ? selectedFile()!.name : 'Selecionar arquivo de vídeo' }}</span>
              <input
                type="file"
                accept=".mp4,.mov,.mkv,.webm,.avi,video/*"
                (change)="onFileSelected($event)"
                style="display: none;"
              />
            </label>

            @if (selectedFile()) {
              <span style="font-size: 0.85rem; color: var(--rv-text-secondary);">
                {{ selectedFile()!.size | formatBytes }}
              </span>
            }

            <button
              type="submit"
              mat-flat-button
              color="primary"
              [disabled]="uploading() || !selectedFile()"
            >
              <mat-icon>cloud_upload</mat-icon>
              {{ uploading() ? 'Enviando vídeo...' : 'Enviar vídeo e criar inspeção' }}
            </button>
          </div>

          @if (uploading()) {
            <mat-progress-bar mode="indeterminate" />
          }

          @if (uploadError()) {
            <div class="rv-banner rv-banner--error" role="alert">
              <mat-icon aria-hidden="true">error_outline</mat-icon>
              <div>{{ uploadError() }}</div>
            </div>
          }
        </form>
      </section>

      <section class="rv-card">
        <div class="rv-card__header">
          <div>
            <h2 class="rv-card__title">Histórico de inspeções</h2>
            <p class="rv-card__subtitle">Filtre por nome ou estado e dispare ou abra uma inspeção.</p>
          </div>

          <button type="button" mat-stroked-button (click)="loadInspections()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            Atualizar lista
          </button>
        </div>

        <div class="rv-filter-bar" style="margin-bottom: 16px;">
          <mat-form-field appearance="outline" style="flex: 1 1 260px;">
            <mat-label>Buscar por título ou arquivo</mat-label>
            <input
              matInput
              [ngModel]="searchTerm()"
              (ngModelChange)="searchTerm.set($event)"
              placeholder="Digite para filtrar..."
            />
          </mat-form-field>

          <mat-form-field appearance="outline" style="width: 210px;">
            <mat-label>Status</mat-label>
            <mat-select [ngModel]="statusFilter()" (ngModelChange)="statusFilter.set($event)">
              <mat-option value="ALL">Todos</mat-option>
              <mat-option value="UPLOADED">Enviado</mat-option>
              <mat-option value="QUEUED">Na fila</mat-option>
              <mat-option value="PROCESSING">Processando</mat-option>
              <mat-option value="COMPLETED">Concluído</mat-option>
              <mat-option value="FAILED">Falhou</mat-option>
            </mat-select>
          </mat-form-field>
        </div>

        @if (loading()) {
          <mat-progress-bar mode="indeterminate" />
        }

        @if (listError()) {
          <div class="rv-banner rv-banner--error" role="alert">
            <mat-icon aria-hidden="true">error_outline</mat-icon>
            <div>{{ listError() }}</div>
          </div>
        } @else if (filteredInspections().length === 0 && !loading()) {
          <rv-empty-state
            icon="search_off"
            title="Nenhuma inspeção encontrada"
            description="Ajuste os filtros ou envie um novo vídeo acima."
          />
        } @else if (filteredInspections().length > 0) {
          <div class="rv-table-wrap">
            <table class="rv-table">
              <thead>
                <tr>
                  <th>Título</th>
                  <th>Arquivo</th>
                  <th>Tamanho</th>
                  <th>Status</th>
                  <th>Criado em</th>
                  <th>Ações</th>
                </tr>
              </thead>
              <tbody>
                @for (item of filteredInspections(); track item.id) {
                  <tr>
                    <td>
                      <strong>{{ item.title }}</strong>
                      @if (item.notes) {
                        <div style="font-size: 0.8rem; color: var(--rv-text-muted);">{{ item.notes }}</div>
                      }
                    </td>
                    <td class="rv-mono">{{ item.originalFilename }}</td>
                    <td>{{ item.fileSizeBytes | formatBytes }}</td>
                    <td><rv-status-badge [status]="item.status" /></td>
                    <td>{{ item.createdAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                    <td>
                      <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                        <a mat-stroked-button [routerLink]="['/inspections', item.id]">
                          Detalhes
                        </a>
                        <button
                          type="button"
                          mat-flat-button
                          color="primary"
                          [disabled]="isActiveStatus(item.status) || triggeringId() === item.id"
                          (click)="triggerProcess(item)"
                        >
                          {{ item.status === 'FAILED' || item.status === 'COMPLETED' ? 'Reprocessar' : 'Processar' }}
                        </button>
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
})
export class InspectionsPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly notifications = inject(NotificationService);
  private readonly router = inject(Router);

  protected readonly title = signal<string>('');
  protected readonly notes = signal<string>('');
  protected readonly selectedFile = signal<File | null>(null);
  protected readonly uploading = signal<boolean>(false);
  protected readonly uploadError = signal<string | null>(null);

  protected readonly loading = signal<boolean>(false);
  protected readonly listError = signal<string | null>(null);
  protected readonly inspections = signal<InspectionSummary[]>([]);
  protected readonly searchTerm = signal<string>('');
  protected readonly statusFilter = signal<InspectionStatus | 'ALL'>('ALL');
  protected readonly triggeringId = signal<string | null>(null);

  protected readonly filteredInspections = computed(() => {
    const query = this.searchTerm().trim().toLowerCase();
    const status = this.statusFilter();
    return this.inspections().filter((item) => {
      const matchesStatus = status === 'ALL' || item.status === status;
      const matchesQuery =
        query.length === 0 ||
        item.title.toLowerCase().includes(query) ||
        item.originalFilename.toLowerCase().includes(query);
      return matchesStatus && matchesQuery;
    });
  });

  ngOnInit(): void {
    this.loadInspections();
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.selectedFile.set(file);
    this.uploadError.set(null);
  }

  protected submitUpload(): void {
    const file = this.selectedFile();
    if (!file) {
      this.uploadError.set('Selecione um arquivo de vídeo antes de enviar.');
      return;
    }
    this.uploading.set(true);
    this.uploadError.set(null);

    this.api.createInspection(file, this.title(), this.notes()).subscribe({
      next: (created) => {
        this.uploading.set(false);
        this.title.set('');
        this.notes.set('');
        this.selectedFile.set(null);
        this.notifications.success('Vídeo validado e inspeção criada com sucesso.');
        this.router.navigate(['/inspections', created.id]);
      },
      error: (err) => {
        const message = extractApiErrorMessage(err, 'Falha ao enviar o vídeo.');
        this.uploading.set(false);
        this.uploadError.set(message);
        this.notifications.error(message);
      },
    });
  }

  protected loadInspections(): void {
    this.loading.set(true);
    this.listError.set(null);
    this.api.listInspections().subscribe({
      next: (list) => {
        this.inspections.set(list);
        this.loading.set(false);
      },
      error: (err) => {
        this.listError.set(extractApiErrorMessage(err, 'Não foi possível carregar as inspeções.'));
        this.loading.set(false);
      },
    });
  }

  protected triggerProcess(item: InspectionSummary): void {
    this.triggeringId.set(item.id);
    this.api.startProcessing(item.id).subscribe({
      next: () => {
        this.triggeringId.set(null);
        this.notifications.info('Processamento assíncrono enfileirado.');
        this.router.navigate(['/inspections', item.id]);
      },
      error: (err) => {
        this.triggeringId.set(null);
        this.notifications.error(
          extractApiErrorMessage(err, 'Não foi possível iniciar o processamento.')
        );
      },
    });
  }

  protected isActiveStatus(status: InspectionStatus): boolean {
    return status === 'QUEUED' || status === 'PROCESSING';
  }
}
