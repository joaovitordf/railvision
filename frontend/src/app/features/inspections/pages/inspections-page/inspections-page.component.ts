import { DatePipe } from '@angular/common';
import { Component, DestroyRef, OnInit, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { InspectionStatus, InspectionSummary } from '../../../../core/models/inspection.models';
import { extractApiErrorMessage } from '../../../../core/services/api-error.util';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { EmptyStateComponent } from '../../../../shared/components/empty-state/empty-state.component';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { FormatBytesPipe } from '../../../../shared/pipes/format-bytes.pipe';
import { FormatDurationPipe } from '../../../../shared/pipes/format-duration.pipe';
import { InspectionsReportsTabComponent } from '../../components/inspections-reports-tab.component';

@Component({
  selector: 'rv-inspections-page',
  standalone: true,
  imports: [
    FormsModule,
    RouterLink,
    DatePipe,
    MatIconModule,
    MatProgressBarModule,
    StatusBadgeComponent,
    EmptyStateComponent,
    FormatBytesPipe,
    FormatDurationPipe,
    InspectionsReportsTabComponent,
  ],
  template: `
    <div class="rv-page">
      <div class="rv-page-header">
        <div>
          <h2 class="rv-page-header__title">Inspeções ferroviárias</h2>
          <p class="rv-page-header__subtitle">
            Importe vídeos ferroviários, acompanhe o processamento e exporte relatórios consolidados.
          </p>
        </div>

        <div class="rv-header-actions">
          <button type="button" class="rv-btn rv-btn--secondary" (click)="loadInspections()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            <span>Atualizar</span>
          </button>
          <button
            type="button"
            class="rv-btn rv-btn--primary"
            data-testid="toggle-new-inspection-btn"
            (click)="showUploadPanel.set(!showUploadPanel())"
          >
            <mat-icon>{{ showUploadPanel() ? 'close' : 'add' }}</mat-icon>
            <span>{{ showUploadPanel() ? 'Fechar envio' : 'Nova inspeção' }}</span>
          </button>
        </div>
      </div>

      @if (showUploadPanel()) {
        <section class="rv-card" data-testid="new-inspection-panel">
          <div class="rv-card__header">
            <div>
              <h3 class="rv-card__title">Nova inspeção por vídeo</h3>
              <p class="rv-card__subtitle">
                Formatos aceitos: MP4, MOV, MKV, WEBM, AVI e OGV. O arquivo original é preservado intacto.
              </p>
            </div>
          </div>

          <form (ngSubmit)="submitUpload()" style="display: flex; flex-direction: column; gap: 16px;">
            <div class="rv-form-grid">
              <div class="rv-field">
                <label class="rv-label" for="insp-title">Título da inspeção (opcional)</label>
                <input
                  id="insp-title"
                  class="rv-input"
                  name="title"
                  [ngModel]="title()"
                  (ngModelChange)="title.set($event)"
                  maxlength="200"
                  placeholder="Ex.: Pátio Norte - Composição 01"
                />
              </div>

              <div class="rv-field">
                <label class="rv-label" for="insp-notes">Observações operacionais (opcional)</label>
                <input
                  id="insp-notes"
                  class="rv-input"
                  name="notes"
                  [ngModel]="notes()"
                  (ngModelChange)="notes.set($event)"
                  maxlength="1000"
                  placeholder="Contexto de câmera, pátio ou iluminação"
                />
              </div>
            </div>

            <div class="rv-dropzone" (click)="videoFileInput.click()">
              <div class="rv-dropzone__left">
                <div class="rv-dropzone__icon">
                  <mat-icon aria-hidden="true">movie</mat-icon>
                </div>
                <div>
                  <div class="rv-dropzone__title">
                    {{ selectedFile() ? selectedFile()!.name : 'Clique para selecionar um arquivo de vídeo' }}
                  </div>
                  <div class="rv-dropzone__hint">
                    @if (selectedFile()) {
                      Tamanho: {{ selectedFile()!.size | formatBytes }}
                    } @else {
                      MP4, WEBM, OGV, MOV, MKV ou AVI · Máximo 500 MB
                    }
                  </div>
                </div>
              </div>

              <input
                #videoFileInput
                type="file"
                accept=".mp4,.mov,.mkv,.webm,.avi,.ogv,video/*"
                (change)="onFileSelected($event)"
                style="display: none;"
              />

              <button
                type="submit"
                class="rv-btn rv-btn--primary"
                [disabled]="uploading() || !selectedFile()"
                (click)="$event.stopPropagation()"
              >
                <mat-icon>cloud_upload</mat-icon>
                <span>{{ uploading() ? 'Enviando vídeo...' : 'Enviar e criar inspeção' }}</span>
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
      }

      <nav class="rv-tabs" role="tablist" aria-label="Visões de inspeções">
        <button
          type="button"
          role="tab"
          class="rv-tab"
          [class.rv-tab--active]="activeTab() === 'list'"
          [attr.aria-selected]="activeTab() === 'list'"
          (click)="switchTab('list')"
        >
          <mat-icon>list_alt</mat-icon>
          <span>Inspeções</span>
          <span class="rv-badge rv-badge--neutral">{{ inspections().length }}</span>
        </button>
        <button
          type="button"
          role="tab"
          class="rv-tab"
          [class.rv-tab--active]="activeTab() === 'reports'"
          [attr.aria-selected]="activeTab() === 'reports'"
          (click)="switchTab('reports')"
        >
          <mat-icon>summarize</mat-icon>
          <span>Relatórios e Exportação</span>
        </button>
      </nav>

      @if (activeTab() === 'list') {
        <section class="rv-card rv-card--table">
          <div class="rv-card__header">
            <div class="rv-filter-bar" style="flex: 1 1 auto;">
              <div class="rv-search">
                <mat-icon aria-hidden="true">search</mat-icon>
                <input
                  class="rv-input"
                  [ngModel]="searchTerm()"
                  (ngModelChange)="searchTerm.set($event)"
                  placeholder="Pesquisar por título ou nome do arquivo..."
                  aria-label="Pesquisar inspeções"
                />
              </div>

              <select
                class="rv-select"
                style="width: 175px;"
                [ngModel]="statusFilter()"
                (ngModelChange)="statusFilter.set($event)"
                aria-label="Filtrar por status"
              >
                <option value="ALL">Status: Todos</option>
                <option value="UPLOADED">Enviado</option>
                <option value="QUEUED">Na fila</option>
                <option value="PROCESSING">Processando</option>
                <option value="COMPLETED">Concluído</option>
                <option value="FAILED">Falhou</option>
              </select>

              <select
                class="rv-select"
                style="width: 175px;"
                [ngModel]="sortOrder()"
                (ngModelChange)="sortOrder.set($event)"
                aria-label="Ordenar por data"
              >
                <option value="newest">Mais recentes</option>
                <option value="oldest">Mais antigas</option>
              </select>
            </div>
          </div>

          @if (loading()) {
            <mat-progress-bar mode="indeterminate" />
          }

          @if (listError()) {
            <div style="padding: 18px 22px;">
              <div class="rv-banner rv-banner--error" role="alert">
                <mat-icon aria-hidden="true">error_outline</mat-icon>
                <div>{{ listError() }}</div>
              </div>
            </div>
          } @else if (filteredInspections().length === 0 && !loading()) {
            <rv-empty-state
              icon="search_off"
              title="Nenhuma inspeção encontrada"
              description="Ajuste os filtros acima ou envie um novo vídeo ferroviário."
            />
          } @else if (filteredInspections().length > 0) {
            <div class="rv-table-wrap">
              <table class="rv-table">
                <thead>
                  <tr>
                    <th>Título</th>
                    <th>Arquivo</th>
                    <th>Criado em</th>
                    <th>Duração</th>
                    <th>Status</th>
                    <th>Vagões</th>
                    <th style="text-align: right;">Ações</th>
                  </tr>
                </thead>
                <tbody>
                  @for (item of filteredInspections(); track item.id) {
                    <tr>
                      <td>
                        <strong>{{ item.title }}</strong>
                        @if (item.notes) {
                          <div style="font-size: 0.78rem; color: var(--rv-text-muted); margin-top: 2px;">{{ item.notes }}</div>
                        }
                      </td>
                      <td class="rv-mono" style="color: var(--rv-text-secondary);">{{ item.originalFilename }}</td>
                      <td style="color: var(--rv-text-secondary);">{{ item.createdAt | date: 'dd/MM/yyyy HH:mm' }}</td>
                      <td class="rv-mono">{{ item.durationMs | formatDuration }}</td>
                      <td><rv-status-badge [status]="item.status" /></td>
                      <td>{{ item.status === 'COMPLETED' ? (item.wagonCount ?? 0) : '—' }}</td>
                      <td style="text-align: right;">
                        <div style="display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end;">
                          <a class="rv-btn rv-btn--secondary rv-btn--sm" [routerLink]="['/inspections', item.id]">
                            Detalhes
                          </a>
                          <button
                            type="button"
                            class="rv-btn rv-btn--primary rv-btn--sm"
                            [disabled]="isActiveStatus(item.status) || triggeringId() === item.id"
                            (click)="triggerProcess(item)"
                          >
                            <mat-icon>play_arrow</mat-icon>
                            <span>{{ item.status === 'FAILED' || item.status === 'COMPLETED' ? 'Reprocessar' : 'Processar' }}</span>
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
      } @else {
        <rv-inspections-reports-tab
          [inspections]="inspections()"
          (refreshRequested)="loadInspections()"
        />
      }
    </div>
  `,
})
export class InspectionsPageComponent implements OnInit {
  private readonly api = inject(RailvisionApiService);
  private readonly notifications = inject(NotificationService);
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly showUploadPanel = signal<boolean>(false);
  protected readonly activeTab = signal<'list' | 'reports'>('list');

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
  protected readonly sortOrder = signal<'newest' | 'oldest'>('newest');
  protected readonly triggeringId = signal<string | null>(null);

  protected readonly filteredInspections = computed(() => {
    const query = this.searchTerm().trim().toLowerCase();
    const status = this.statusFilter();
    const list = this.inspections().filter((item) => {
      const matchesStatus = status === 'ALL' || item.status === status;
      const matchesQuery =
        query.length === 0 ||
        item.title.toLowerCase().includes(query) ||
        item.originalFilename.toLowerCase().includes(query);
      return matchesStatus && matchesQuery;
    });
    const order = this.sortOrder();
    return [...list].sort((a, b) => {
      const diff = new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
      return order === 'newest' ? diff : -diff;
    });
  });

  ngOnInit(): void {
    this.route.queryParamMap?.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((params) => {
      if (params.get('tab') === 'reports') {
        this.activeTab.set('reports');
      } else if (params.get('tab') === 'list') {
        this.activeTab.set('list');
      }
      if (params.get('action') === 'new') {
        this.showUploadPanel.set(true);
      }
    });
    this.loadInspections();
  }

  protected switchTab(tab: 'list' | 'reports'): void {
    this.activeTab.set(tab);
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { tab: tab === 'reports' ? 'reports' : null },
      queryParamsHandling: 'merge',
    });
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    this.selectedFile.set(file);
    this.uploadError.set(null);
    input.value = '';
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
