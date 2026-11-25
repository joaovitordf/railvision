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
import {
  InspectionDetailPageComponent,
  computeContainedVideoRect,
} from './inspection-detail-page.component';

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
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="demo-fixture-banner"]')).toBeTruthy();

    const wagonsTabBtn = el.querySelector('[data-testid="tab-wagons"]') as HTMLButtonElement;
    wagonsTabBtn.click();
    fixture.detectChanges();

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

  it('computes object-fit contain rect for letterboxed/pillarboxed videos', () => {
    const exact169 = computeContainedVideoRect(1600, 900, 1920, 1080);
    expect(exact169.offsetX).toBeCloseTo(0, 2);
    expect(exact169.offsetY).toBeCloseTo(0, 2);
    expect(exact169.drawWidth).toBeCloseTo(1600, 2);
    expect(exact169.drawHeight).toBeCloseTo(900, 2);

    const pillarboxed = computeContainedVideoRect(1600, 900, 1200, 900);
    expect(pillarboxed.drawHeight).toBeCloseTo(900, 2);
    expect(pillarboxed.drawWidth).toBeCloseTo(1200, 2);
    expect(pillarboxed.offsetX).toBeCloseTo(200, 2);
    expect(pillarboxed.offsetY).toBeCloseTo(0, 2);

    const letterboxed = computeContainedVideoRect(800, 600, 2560, 1080);
    expect(letterboxed.drawWidth).toBeCloseTo(800, 2);
    expect(letterboxed.drawHeight).toBeCloseTo(337.5, 2);
    expect(letterboxed.offsetX).toBeCloseTo(0, 2);
    expect(letterboxed.offsetY).toBeCloseTo(131.25, 2);
  });

  it('renders real non-demo inspection with not_evaluated loading/material and clears wagons on reprocess', async () => {
    const realDetail: InspectionDetail = {
      ...baseDetail,
      id: 'insp-real',
      wagonCount: 1,
      latestJob: {
        ...baseDetail.latestJob!,
        engineType: 'VISION_DETECTION_ONLY',
        isDemoSimulation: false,
      },
    };
    const realWagon: WagonResult = {
      ...sampleWagons[0],
      id: 'wagon-real-1',
      inspectionId: 'insp-real',
      material: null,
      loadingEstimate: {
        level: 'not_evaluated',
        levelObservationSpread: ['not_evaluated'],
        estimatedFillPercentage: null,
        confidence: null,
        measurementType: 'not_evaluated',
        physicalVolumeVerified: false,
      },
      consolidation: { observationsCount: 3, disagreements: [], method: 'bytetrack_median' },
      qualityFlags: ['short_track_2_obs'],
      reviewRequired: true,
      reviewReasons: ['low_observation_count'],
      isExample: false,
    };

    const apiMock = {
      getInspection: vi.fn().mockReturnValue(of(realDetail)),
      listWagons: vi.fn().mockReturnValue(of([realWagon])),
      listObservations: vi.fn().mockReturnValue(of(sampleObservations)),
      startProcessing: vi.fn().mockReturnValue(of({ ...realDetail.latestJob!, status: 'QUEUED' })),
      getJob: vi.fn().mockReturnValue(of({ ...realDetail.latestJob!, status: 'QUEUED' })),
      mediaUrl: vi.fn().mockReturnValue('/api/v1/inspections/insp-real/media'),
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
          useValue: { snapshot: { paramMap: convertToParamMap({ id: 'insp-real' }) } },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(InspectionDetailPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="demo-fixture-banner"]')).toBeNull();

    const wagonsTabBtn = el.querySelector('[data-testid="tab-wagons"]') as HTMLButtonElement;
    wagonsTabBtn.click();
    fixture.detectChanges();

    expect(el.textContent).toContain('Não avaliado');
    expect(el.textContent).toContain('bytetrack_median');
    expect(el.textContent).toContain('short_track_2_obs');

    const comp = fixture.componentInstance as unknown as {
      startOrReprocess: () => void;
      wagons: () => WagonResult[];
    };
    comp.startOrReprocess();
    expect(comp.wagons().length).toBe(0);
  });
});
