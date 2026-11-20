import { TestBed } from '@angular/core/testing';
import { ActivatedRoute, convertToParamMap, provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { vi } from 'vitest';
import {
  InspectionDetail,
  WagonObservation,
  WagonResult,
} from '../../../../core/models/inspection.models';
import { NotificationService } from '../../../../core/services/notification.service';
import { RailvisionApiService } from '../../../../core/services/railvision-api.service';
import { InspectionDetailPageComponent } from './inspection-detail-page.component';

describe('InspectionDetailPageComponent', () => {
  const baseDetail: InspectionDetail = {
    id: 'insp-123',
    title: 'Inspeção Teste',
    notes: 'Câmera elevada',
    status: 'COMPLETED',
    originalFilename: 'demo.mp4',
    contentType: 'video/mp4',
    fileSizeBytes: 2048,
    sha256Checksum: 'abcdef1234567890',
    latestJobId: 'job-1',
    durationMs: null,
    widthPx: null,
    heightPx: null,
    fps: null,
    codec: null,
    wagonCount: 2,
    reviewRequiredCount: 1,
    latestJob: {
      id: 'job-1',
      inspectionId: 'insp-123',
      status: 'COMPLETED',
      progressPercent: 100,
      engineType: 'DEMO_FIXTURE_ENGINE',
      isDemoSimulation: true,
      errorCode: null,
      errorMessage: null,
      startedAt: '2026-11-19T00:00:01Z',
      finishedAt: '2026-11-19T00:00:02Z',
      createdAt: '2026-11-19T00:00:00Z',
    },
    createdAt: '2026-11-19T00:00:00Z',
    updatedAt: '2026-11-19T00:00:02Z',
  };

  const sampleWagons: WagonResult[] = [
    {
      id: 'wagon-1',
      inspectionId: 'insp-123',
      jobId: 'job-1',
      wagonIdentity: { trackId: 'example-track-001', sequenceIndex: 1, physicalId: null },
      material: {
        materialClass: 'coal',
        confidence: 0,
        voteDistribution: { coal: 1, iron_ore: 1 },
        rejectedAsUnknown: false,
      },
      loadingEstimate: {
        level: 'medium',
        levelObservationSpread: ['medium', 'low'],
        estimatedFillPercentage: null,
        confidence: 0,
        measurementType: 'visual_estimate',
        physicalVolumeVerified: false,
      },
      modelVersions: { detector: 'demo@0.0.0' },
      consolidation: {
        observationsCount: 1,
        disagreements: ['Divergência de exemplo'],
        method: 'demo_fixture_only',
      },
      qualityFlags: ['example_fixture'],
      reviewRequired: true,
      reviewReasons: ['observation_disagreement'],
      isExample: true,
      createdAt: '2026-11-19T00:00:02Z',
    },
    {
      id: 'wagon-2',
      inspectionId: 'insp-123',
      jobId: 'job-1',
      wagonIdentity: { trackId: 'example-track-002', sequenceIndex: 2, physicalId: null },
      material: null,
      loadingEstimate: {
        level: 'empty',
        levelObservationSpread: ['empty'],
        estimatedFillPercentage: null,
        confidence: 0,
        measurementType: 'visual_estimate',
        physicalVolumeVerified: false,
      },
      modelVersions: { detector: 'demo@0.0.0' },
      consolidation: { observationsCount: 1, disagreements: [], method: 'demo_fixture_only' },
      qualityFlags: ['example_fixture'],
      reviewRequired: false,
      reviewReasons: [],
      isExample: true,
      createdAt: '2026-11-19T00:00:02Z',
    },
  ];

  const sampleObservations: WagonObservation[] = [
    {
      id: 'obs-1',
      wagonResultId: 'wagon-1',
      frameIndex: 0,
      frameTimestampMs: 0,
      boundingBox: { x: 0.1, y: 0.2, width: 0.3, height: 0.4, coordinateSystem: 'normalized' },
      detectedClass: 'freight_wagon_open',
      detectionConfidence: 0,
      materialClass: 'coal',
      materialConfidence: 0,
      loadingLevel: 'medium',
      loadingConfidence: 0,
      isExample: true,
    },
  ];

  it('renders demo fixture banner, review badge, and N/A (vagão vazio) for empty wagon', async () => {
    const apiMock = {
      getInspection: vi.fn().mockReturnValue(of(baseDetail)),
      listWagons: vi.fn().mockReturnValue(of(sampleWagons)),
      listObservations: vi.fn().mockReturnValue(of(sampleObservations)),
      startProcessing: vi.fn(),
      getJob: vi.fn(),
      mediaUrl: vi.fn().mockReturnValue('/api/v1/inspections/insp-123/media'),
    };

    await TestBed.configureTestingModule({
      imports: [InspectionDetailPageComponent],
      providers: [
        provideRouter([]),
        { provide: RailvisionApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: convertToParamMap({ id: 'insp-123' }) } },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InspectionDetailPageComponent);
    await fixture.whenStable();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="demo-fixture-banner"]')).toBeTruthy();
    expect(el.querySelector('[data-testid="review-required-badge"]')?.textContent).toContain(
      'Revisão necessária'
    );
    expect(el.textContent).toContain('N/A (vagão vazio)');
    expect(el.textContent).toContain('Vazio');
  });

  it('renders sanitized job error banner when latest job failed', async () => {
    const failedDetail: InspectionDetail = {
      ...baseDetail,
      status: 'FAILED',
      wagonCount: 0,
      reviewRequiredCount: 0,
      latestJob: {
        ...baseDetail.latestJob!,
        status: 'FAILED',
        errorCode: 'PROCESS_INTERRUPTED',
        errorMessage: 'O processamento foi interrompido pelo encerramento do servidor.',
      },
    };

    const apiMock = {
      getInspection: vi.fn().mockReturnValue(of(failedDetail)),
      listWagons: vi.fn().mockReturnValue(of([])),
      listObservations: vi.fn().mockReturnValue(of([])),
      startProcessing: vi.fn(),
      getJob: vi.fn(),
      mediaUrl: vi.fn().mockReturnValue('/api/v1/inspections/insp-123/media'),
    };

    await TestBed.configureTestingModule({
      imports: [InspectionDetailPageComponent],
      providers: [
        provideRouter([]),
        { provide: RailvisionApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
        {
          provide: ActivatedRoute,
          useValue: { snapshot: { paramMap: convertToParamMap({ id: 'insp-123' }) } },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InspectionDetailPageComponent);
    await fixture.whenStable();

    const el = fixture.nativeElement as HTMLElement;
    const failedBanner = el.querySelector('[data-testid="job-failed-banner"]');
    expect(failedBanner).toBeTruthy();
    expect(failedBanner?.textContent).toContain('PROCESS_INTERRUPTED');
    expect(el.textContent).toContain('Reprocessar');
  });
});
