import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { InspectionSummary } from '../../../../core/models/inspection.models';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { AnnotationProjectSummary } from '../../../annotations/models/annotations.models';
import { AnnotationsApiService } from '../../../annotations/services/annotations-api.service';
import { RunListItem } from '../../../lab/models/lab.models';
import { LabApiService } from '../../../lab/services/lab-api.service';
import { OverviewPageComponent } from './overview-page.component';

describe('OverviewPageComponent', () => {
  it('renders KPI counts, recent inspections, and refreshes on button click', async () => {
    const inspections: InspectionSummary[] = [
      {
        id: 'insp-1',
        title: 'Composição Carvão Norte',
        notes: null,
        originalFilename: 'trem_01.mp4',
        contentType: 'video/mp4',
        fileSizeBytes: 1024 * 1024,
        sha256Checksum: 'a'.repeat(64),
        latestJobId: 'job-1',
        status: 'COMPLETED',
        durationMs: 12000,
        wagonCount: 14,
        createdAt: '2026-11-24T18:00:00Z',
        updatedAt: '2026-11-24T18:05:00Z',
      },
      {
        id: 'insp-2',
        title: 'Pátio Sul Falha',
        notes: null,
        originalFilename: 'trem_02.mp4',
        contentType: 'video/mp4',
        fileSizeBytes: 512 * 1024,
        sha256Checksum: 'b'.repeat(64),
        latestJobId: null,
        status: 'FAILED',
        durationMs: 4000,
        wagonCount: undefined,
        createdAt: '2026-11-24T17:00:00Z',
        updatedAt: '2026-11-24T17:02:00Z',
      },
    ];
    const labRuns: RunListItem[] = [
      {
        runId: 'lab-1',
        mediaType: 'IMAGE',
        status: 'COMPLETED',
        createdAt: '2026-11-24T19:00:00Z',
        originalFilename: 'frame_01.jpg',
        modelName: 'exp04c_yolo26n_opentop_640',
        deviceType: 'GPU',
        activeExecutionProvider: 'PyTorch CUDA',
        totalDetections: 3,
        noDetections: false,
        uniqueTrackCount: null,
        processingFps: 45,
        avgFrameMs: 22,
        totalMs: 22,
        trackingIntegrated: false,
      },
    ];
    const projects: AnnotationProjectSummary[] = [
      {
        id: 'proj-1',
        name: 'WS2 Oberhausen',
        campaignId: 'camp_oberhausen',
        sourceOrigin: 'wikimedia_commons',
        description: null,
        classes: ['open_top_freight_wagon'],
        itemCount: 18,
        videoCount: 1,
        statusCounts: { pre_annotated: 18 },
        reviewedCount: 0,
        approvedCount: 0,
        createdAt: '2026-11-24T16:00:00Z',
        updatedAt: '2026-11-24T19:30:00Z',
      },
    ];

    const railvisionApiMock = { listInspections: vi.fn().mockReturnValue(of(inspections)) };
    const labApiMock = { listRuns: vi.fn().mockReturnValue(of(labRuns)) };
    const annotationsApiMock = { listProjects: vi.fn().mockReturnValue(of(projects)) };

    await TestBed.configureTestingModule({
      imports: [OverviewPageComponent],
      providers: [
        provideRouter([]),
        { provide: RailvisionApiService, useValue: railvisionApiMock },
        { provide: LabApiService, useValue: labApiMock },
        { provide: AnnotationsApiService, useValue: annotationsApiMock },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(OverviewPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    const kpiValues = Array.from(el.querySelectorAll('.rv-kpi-card__value')).map((n) => n.textContent?.trim());
    expect(kpiValues).toEqual(['2', '1', '0', '1']);
    expect(el.textContent).toContain('Composição Carvão Norte');
    expect(el.textContent).toContain('WS2 Oberhausen');

    const refreshBtn = Array.from(el.querySelectorAll('button')).find((b) =>
      b.textContent?.includes('Atualizar')
    ) as HTMLButtonElement;
    expect(refreshBtn).toBeTruthy();
    refreshBtn.click();
    expect(railvisionApiMock.listInspections).toHaveBeenCalledTimes(2);
  });
});
