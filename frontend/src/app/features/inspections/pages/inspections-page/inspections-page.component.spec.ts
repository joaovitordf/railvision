import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { InspectionSummary } from '../../../../core/models/inspection.models';
import { InspectionReportExportService } from '../../../../core/services/inspection-report-export.service';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { InspectionsPageComponent } from './inspections-page.component';

describe('InspectionsPageComponent', () => {
  it('toggles upload drawer, filters list, switches to Reports tab, and triggers processing', async () => {
    const items: InspectionSummary[] = [
      {
        id: '11111111-1111-1111-1111-111111111111',
        title: 'Composição Alfa',
        notes: 'Pátio A',
        originalFilename: 'alfa.mp4',
        contentType: 'video/mp4',
        fileSizeBytes: 2048,
        sha256Checksum: 'a'.repeat(64),
        latestJobId: 'job-alfa',
        status: 'COMPLETED',
        durationMs: 8500,
        wagonCount: 9,
        createdAt: '2026-11-24T12:00:00Z',
        updatedAt: '2026-11-24T12:01:00Z',
      },
      {
        id: '22222222-2222-2222-2222-222222222222',
        title: 'Composição Beta',
        notes: null,
        originalFilename: 'beta.mp4',
        contentType: 'video/mp4',
        fileSizeBytes: 1024,
        sha256Checksum: 'b'.repeat(64),
        latestJobId: null,
        status: 'UPLOADED',
        durationMs: 3000,
        wagonCount: undefined,
        createdAt: '2026-11-24T11:00:00Z',
        updatedAt: '2026-11-24T11:00:00Z',
      },
    ];

    const apiMock = {
      listInspections: vi.fn().mockReturnValue(of(items)),
      createInspection: vi.fn(),
      startProcessing: vi.fn().mockReturnValue(of({ id: 'job-1', status: 'QUEUED' })),
    };

    await TestBed.configureTestingModule({
      imports: [InspectionsPageComponent],
      providers: [
        provideRouter([]),
        { provide: RailvisionApiService, useValue: apiMock },
        { provide: InspectionReportExportService, useValue: { exportSummaryById: vi.fn() } },
        { provide: NotificationService, useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() } },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InspectionsPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('tbody tr').length).toBe(2);

    // Toggle New Inspection panel open and closed
    const toggleBtn = el.querySelector('[data-testid="toggle-new-inspection-btn"]') as HTMLButtonElement;
    toggleBtn.click();
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="new-inspection-panel"]')).toBeTruthy();
    toggleBtn.click();
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="new-inspection-panel"]')).toBeNull();

    // Filter by search input
    const searchInput = el.querySelector('input[aria-label="Pesquisar inspeções"]') as HTMLInputElement;
    searchInput.value = 'Beta';
    searchInput.dispatchEvent(new Event('input'));
    fixture.detectChanges();
    expect(el.querySelectorAll('tbody tr').length).toBe(1);
    expect(el.textContent).toContain('Composição Beta');

    // Trigger Process button on filtered row
    const procBtn = el.querySelector('tbody button') as HTMLButtonElement;
    procBtn.click();
    expect(apiMock.startProcessing).toHaveBeenCalledWith('22222222-2222-2222-2222-222222222222');

    // Switch to Reports tab
    const tabs = Array.from(el.querySelectorAll('button[role="tab"]')) as HTMLButtonElement[];
    tabs[1].click();
    fixture.detectChanges();
    expect(el.textContent).toContain('Relatórios de inspeções concluídas');
    expect(el.textContent).toContain('Composição Alfa');
  });
});
