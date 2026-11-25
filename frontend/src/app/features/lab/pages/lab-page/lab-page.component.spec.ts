import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { NotificationService } from '../../../../core/services/notification.service';
import { ImageRunResult, RuntimeStatus, VideoRunStatus } from '../../models/lab.models';
import { TrainingOverview } from '../../models/training.models';
import { LabApiService } from '../../services/lab-api.service';
import { TrainingApiService } from '../../services/training-api.service';
import { LabPageComponent } from './lab-page.component';

describe('LabPageComponent', () => {
  const gpuStatus: RuntimeStatus = {
    available: true,
    unavailableReason: null,
    modelName: 'exp04c_yolo26n_opentop_640',
    sourceExperiment: 'exp04c_yolo26n_opentop_freeze10',
    modelDisplayPath: 'artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt',
    modelSha256: '8ade6feb99abe63d94db',
    inputSize: 640,
    inputShape: [1, 3, 640, 640],
    outputShape: [1, 5, 8400],
    classNames: { '0': 'open_top_freight_wagon' },
    requestedExecutionProvider: 'pytorch_cuda',
    activeExecutionProvider: 'PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)',
    deviceType: 'GPU',
    cudaAvailable: true,
    fallbackOccurred: false,
    availableProviders: ['CPUExecutionProvider'],
    deviceDiagnostic: 'Runtime padrão PyTorch CUDA ativo em GPU (NVIDIA GeForce RTX 4060 Ti).',
    trackingIntegrated: true,
    trackingStatus: 'OPTIONAL_BYTETRACK',
    trackingNotice: 'Tracking via ByteTrack disponível como opção habilitável na análise de vídeo.',
    limitationsNotice: 'Modelo PoC exp04c não aprovado para produção.',
    defaultConfThreshold: 0.25,
    defaultIouThreshold: 0.45,
    maxVideoFramesPerUiRun: 900,
    supportedImageExtensions: ['jpg', 'png'],
    supportedVideoExtensions: ['mp4', 'webm'],
    backendEngine: 'python-fastapi',
    defaultRuntime: 'pytorch_cuda',
    availableRuntimes: [
      {
        id: 'pytorch_cuda',
        label: 'PyTorch + Ultralytics CUDA (Padrão)',
        deviceType: 'GPU',
        available: true,
        provider: 'PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)',
        diagnostic: 'CUDA ativo.',
      },
      {
        id: 'onnx_cpu',
        label: 'ONNX Runtime CPU',
        deviceType: 'CPU',
        available: true,
        provider: 'CPUExecutionProvider',
        diagnostic: 'ONNX CPU.',
      },
    ],
    defaultOutputResolution: '1080p',
    availableOutputResolutions: ['1080p', 'native'],
    videoEncoder: 'libx264',
    videoEncoderCrf: 23,
    trackingAlgorithms: ['none', 'bytetrack'],
  };

  it('renders live GPU provider status and zero-detection banner in Inference tab', async () => {
    const emptyImageResult: ImageRunResult = {
      runId: 'run-empty-1',
      mediaType: 'IMAGE',
      status: 'COMPLETED',
      createdAt: '2026-11-20T19:00:00Z',
      originalFilename: 'snow_negative.png',
      imageWidth: 1280,
      imageHeight: 720,
      detectionCount: 0,
      noDetections: true,
      emptyMessage: 'Nenhum objeto detectado com confiança >= 0.25.',
      detections: [],
      metrics: {
        preprocessMs: 4.5,
        inferenceMs: 6.2,
        postprocessMs: 0.4,
        decodeOrEncodeMs: 8.1,
        totalMs: 19.2,
        avgFrameMs: 11.1,
        p50FrameMs: 11.1,
        p95FrameMs: 11.1,
        processingFps: 52.0,
        inferenceOnlyFps: 161.2,
      },
      modelConfig: {
        modelName: 'exp04c_yolo26n_opentop_640',
        sourceExperiment: 'exp04c_yolo26n_opentop_freeze10',
        modelSha256: '8ade6feb99abe63d94db',
        inputSize: 640,
        confThreshold: 0.25,
        iouThreshold: 0.45,
        sampleFps: null,
        deviceType: 'GPU',
        activeExecutionProvider: 'PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)',
        trackingIntegrated: false,
        trackingStatus: 'NOT_ENABLED_IN_RUN',
        exploratoryNotice: 'Execução exploratória local.',
        runtimeId: 'pytorch_cuda',
      },
      files: {
        originalUrl: '/api/v1/vision-lab/runs/run-empty-1/files/original',
        annotatedUrl: '/api/v1/vision-lab/runs/run-empty-1/files/annotated',
        originalDownloadUrl: '/api/v1/vision-lab/runs/run-empty-1/files/original?download=true',
        annotatedDownloadUrl: '/api/v1/vision-lab/runs/run-empty-1/files/annotated?download=true',
        originalFilename: 'original.png',
        annotatedFilename: 'annotated.png',
        annotatedContentType: 'image/png',
      },
    };

    const apiMock = {
      getStatus: vi.fn().mockReturnValue(of(gpuStatus)),
      listRuns: vi.fn().mockReturnValue(of([])),
      getRun: vi.fn().mockReturnValue(of(emptyImageResult)),
      analyzeImage: vi.fn().mockReturnValue(of(emptyImageResult)),
      analyzeVideo: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [LabPageComponent],
      providers: [
        provideRouter([]),
        { provide: LabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(LabPageComponent);
    await fixture.whenStable();

    const component = fixture.componentInstance;
    component.activeRun.set(emptyImageResult);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('NVIDIA GeForce RTX 4060 Ti');
    expect(el.querySelector('[data-testid="no-detections-banner"]')?.textContent).toContain(
      'Nenhum objeto detectado'
    );
    expect(el.querySelector('[data-testid="kpi-detections"]')?.textContent?.trim()).toBe('0');
  });

  it('supports runtime selection, 1080p libx264 output metadata, and ByteTrack badge on video runs', async () => {
    const trackedVideo: VideoRunStatus = {
      runId: 'run-vid-py-1',
      mediaType: 'VIDEO',
      status: 'COMPLETED',
      createdAt: '2026-11-20T20:00:00Z',
      startedAt: '2026-11-20T20:00:00Z',
      completedAt: '2026-11-20T20:00:02Z',
      originalFilename: 'vid_15.ogv',
      videoWidth: 1920,
      videoHeight: 1080,
      sourceFps: 25,
      sourceCodec: 'theora',
      sourceDurationMs: 10000,
      outputCodec: 'h264',
      outputContainer: 'mp4',
      browserPlayable: true,
      estimatedTotalFrames: 10,
      processedFrames: 10,
      progressPercent: 100,
      framesWithDetections: 8,
      totalDetections: 16,
      noDetections: false,
      truncatedByFrameLimit: false,
      emptyMessage: null,
      errorMessage: null,
      metrics: {
        preprocessMs: 3.1,
        inferenceMs: 6.4,
        postprocessMs: 0.3,
        decodeOrEncodeMs: 8.2,
        totalMs: 420.0,
        avgFrameMs: 18.0,
        p50FrameMs: 17.5,
        p95FrameMs: 22.1,
        processingFps: 23.8,
        inferenceOnlyFps: 156.2,
      },
      modelConfig: {
        modelName: 'exp04c_yolo26n_opentop_640',
        sourceExperiment: 'exp04c_yolo26n_opentop_freeze10',
        modelSha256: 'abc123456789',
        inputSize: 640,
        confThreshold: 0.25,
        iouThreshold: 0.45,
        sampleFps: 5,
        deviceType: 'GPU',
        activeExecutionProvider: 'PyTorch CUDA (NVIDIA GeForce RTX 4060 Ti)',
        trackingIntegrated: true,
        trackingStatus: 'BYTETRACK_ENABLED',
        exploratoryNotice: 'Execução exploratória local.',
        runtimeId: 'pytorch_cuda',
        outputResolution: '1080p',
        trackerAlgorithm: 'bytetrack',
      },
      files: {
        originalUrl: '/api/v1/vision-lab/runs/run-vid-py-1/files/original',
        annotatedUrl: '/api/v1/vision-lab/runs/run-vid-py-1/files/annotated',
        originalDownloadUrl: '/api/v1/vision-lab/runs/run-vid-py-1/files/original?download=true',
        annotatedDownloadUrl: '/api/v1/vision-lab/runs/run-vid-py-1/files/annotated?download=true',
        originalFilename: 'original.ogv',
        annotatedFilename: 'annotated.mp4',
        annotatedContentType: 'video/mp4',
      },
      frameSummaries: [],
      outputResolution: '1080p',
      outputWidth: 1920,
      outputHeight: 1080,
      outputEncoder: 'libx264',
      outputCrf: 23,
      uniqueTrackCount: 3,
    };

    const apiMock = {
      getStatus: vi.fn().mockReturnValue(of(gpuStatus)),
      listRuns: vi.fn().mockReturnValue(of([])),
      getRun: vi.fn().mockReturnValue(of(trackedVideo)),
      analyzeImage: vi.fn(),
      analyzeVideo: vi.fn().mockReturnValue(of(trackedVideo)),
    };

    await TestBed.configureTestingModule({
      imports: [LabPageComponent],
      providers: [
        provideRouter([]),
        { provide: LabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(LabPageComponent);
    await fixture.whenStable();

    const component = fixture.componentInstance;
    expect(component.selectedRuntime()).toBe('pytorch_cuda');
    component.activeRun.set(trackedVideo);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('PyTorch CUDA');
    expect(el.querySelector('[data-testid="runtime-select"]')).toBeTruthy();
    expect(el.querySelector('[data-testid="video-output-badge"]')?.textContent).toContain('libx264');
    expect(el.querySelector('[data-testid="video-tracking-badge"]')?.textContent).toContain('3 tracks únicos');
  });

  it('auto-selects CPU fallback runtime when CUDA is unavailable', async () => {
    const cpuOnlyStatus: RuntimeStatus = {
      ...gpuStatus,
      available: false,
      unavailableReason: 'CUDA indisponível no host para pytorch_cuda.',
      cudaAvailable: false,
      deviceType: 'UNAVAILABLE',
      activeExecutionProvider: 'CUDAUnavailable',
      availableRuntimes: [
        {
          id: 'pytorch_cuda',
          label: 'PyTorch CUDA',
          deviceType: 'UNAVAILABLE',
          available: false,
          provider: 'CUDAExecutionUnavailable',
          diagnostic: 'CUDA indisponível.',
        },
        {
          id: 'onnx_cpu',
          label: 'ONNX Runtime CPU',
          deviceType: 'CPU',
          available: true,
          provider: 'CPUExecutionProvider',
          diagnostic: 'ONNX CPU pronto.',
        },
      ],
    };

    const apiMock = {
      getStatus: vi.fn().mockReturnValue(of(cpuOnlyStatus)),
      listRuns: vi.fn().mockReturnValue(of([])),
      getRun: vi.fn(),
      analyzeImage: vi.fn(),
      analyzeVideo: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [LabPageComponent],
      providers: [
        provideRouter([]),
        { provide: LabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(LabPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const comp = fixture.componentInstance;
    expect(comp.selectedRuntime()).toBe('onnx_cpu');
    comp.selectedFile.set(new File(['dummy'], 'sample.png', { type: 'image/png' }));
    expect(comp.canSubmit()).toBe(true);
    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('[data-testid="allow-cpu-fallback-checkbox"]')).toBeTruthy();
  });

  it('supports Model Registry tab, A/B compare warnings tab, and HITL frame curation link', async () => {
    const mockRuns = [
      {
        runId: 'run-a',
        mediaType: 'VIDEO' as const,
        status: 'COMPLETED',
        originalFilename: 'vid_15.ogv',
        createdAt: '2026-11-24T00:00:00Z',
        totalDetections: 12,
        noDetections: false,
        processingFps: 25.0,
        avgFrameMs: 14.0,
        totalMs: 400.0,
        deviceType: 'GPU',
        activeExecutionProvider: 'PyTorch CUDA',
        modelName: 'exp04c_yolo26n_opentop_640',
      },
      {
        runId: 'run-b',
        mediaType: 'VIDEO' as const,
        status: 'COMPLETED',
        originalFilename: 'vid_15.ogv',
        createdAt: '2026-11-24T00:01:00Z',
        totalDetections: 15,
        noDetections: false,
        processingFps: 22.0,
        avgFrameMs: 16.0,
        totalMs: 450.0,
        deviceType: 'GPU',
        activeExecutionProvider: 'PyTorch CUDA',
        modelName: 'exp04a_yolo26n_opentop_from_coco',
      },
    ];
    const apiMock = {
      getStatus: vi.fn().mockReturnValue(of(gpuStatus)),
      listCheckpoints: vi.fn().mockReturnValue(
        of([
          {
            id: 'exp04c_yolo26n_opentop_640',
            modelName: 'exp04c_yolo26n_opentop_640',
            sourceExperiment: 'exp04c_yolo26n_opentop_freeze10',
            architecture: 'yolo26n',
            status: 'active',
            inputSize: 640,
            classNames: { '0': 'open_top_freight_wagon' },
            pytorchWeightsPath: 'artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt',
            pytorchWeightsSha256: 'abc12345678900',
            onnxModelPath: null,
            onnxMetaPath: null,
            onnxModelSha256: null,
            limitationsNotice: 'PoC',
            valMetrics: { map50: 0.901 },
            testMetrics: { map50: 0.2553 },
          },
        ])
      ),
      listRuns: vi.fn().mockReturnValue(of(mockRuns)),
      getRun: vi.fn(),
      analyzeImage: vi.fn(),
      analyzeVideo: vi.fn(),
      compareRuns: vi.fn().mockReturnValue(
        of({
          runA: {
            runId: 'run-a',
            mediaType: 'VIDEO',
            status: 'COMPLETED',
            originalFilename: 'vid_15.ogv',
            modelName: 'exp04c_yolo26n_opentop_640',
            sourceExperiment: 'exp04c',
            deviceType: 'GPU',
            totalDetections: 12,
            processingFps: 25.0,
            avgFrameMs: 14.0,
          },
          runB: {
            runId: 'run-b',
            mediaType: 'VIDEO',
            status: 'COMPLETED',
            originalFilename: 'vid_15.ogv',
            modelName: 'exp04a_yolo26n_opentop_from_coco',
            sourceExperiment: 'exp04a',
            deviceType: 'GPU',
            totalDetections: 15,
            processingFps: 22.0,
            avgFrameMs: 16.0,
          },
          deltaDetections: 3,
          comparable: false,
          comparabilityWarnings: ['Limiares diferentes (Run A conf=0.25 vs Run B conf=0.55).'],
          deltaProcessingFps: -3.0,
          deltaAvgFrameMs: 2.0,
          frameDeltas: [],
        })
      ),
      curateFrame: vi.fn().mockReturnValue(
        of({
          runId: 'run-vid-py-1',
          frameIndex: 0,
          imagePath: 'data/processed/curation-inbox/2026-11-24/run-vid_f000000.jpg',
          labelPath: 'data/processed/curation-inbox/2026-11-24/run-vid_f000000.txt',
          metadataPath: 'data/processed/curation-inbox/2026-11-24/run-vid_f000000.meta.json',
          sourceSha256: 'deadbeef',
          reservedBenchmarkWarning: true,
          detectionCount: 1,
          reason: 'false_negative',
          humanReviewed: false,
          autoTrainEligible: false,
          createdAt: '2026-11-24T00:02:00Z',
        })
      ),
    };

    await TestBed.configureTestingModule({
      imports: [LabPageComponent],
      providers: [
        provideRouter([]),
        { provide: LabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(LabPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const comp = fixture.componentInstance;
    const el = fixture.nativeElement as HTMLElement;
    expect(comp.checkpoints().length).toBe(1);

    // Check Compare tab
    (el.querySelector('[data-testid="lab-tab-compare"]') as HTMLButtonElement).click();
    comp.handleCompare('run-a', 'run-b');
    fixture.detectChanges();
    expect(comp.comparison()?.comparable).toBe(false);
    expect(el.querySelector('[data-testid="compare-warnings-banner"]')?.textContent).toContain('Limiares diferentes');

    // Check Models Registry tab
    (el.querySelector('[data-testid="lab-tab-models"]') as HTMLButtonElement).click();
    fixture.detectChanges();
    expect(el.textContent).toContain('Model Registry');
    expect(el.textContent).toContain('0.901');
  });

  it('renders Training tab with chained version lineage, dataset warning, and start run action', async () => {
    const trainingOverview: TrainingOverview = {
      lineageName: 'Detector de Vagões Abertos',
      baselineModelId: 'exp04c_yolo26n_opentop_640',
      baselineMetrics: {
        modelId: 'exp04c_yolo26n_opentop_640',
        versionLabel: 'v0',
        sourceExperiment: 'exp04c_yolo26n_opentop_freeze10',
        map50: 0.9218,
        map5095: 0.6266,
        precision: 0.8394,
        recall: 0.9127,
      },
      nextVersionNumber: 1,
      nextVersionLabel: 'v1',
      recommendedBaseId: 'checkpoint:exp04c_yolo26n_opentop_640',
      recommendedBaseLabel: 'Baseline inicial (v0 · exp04c_yolo26n_opentop_640)',
      activeRunId: null,
      gpuAvailable: true,
      gpuName: 'NVIDIA GeForce RTX 4060 Ti',
      datasets: [
        {
          id: 'open-top-yolo-v1',
          name: 'open-top-yolo-v1',
          dataYamlPath: 'data/processed/open-top-yolo-v1/data.yaml',
          trainCount: 82,
          valCount: 24,
          valCampaigns: ['camp_sishen_saldanha'],
          valFingerprint: 'abc12345',
          hasFrozenTest: true,
          warning: 'Validação atual possui apenas 1 campanha (camp_sishen_saldanha).',
        },
      ],
      lineagePoints: [
        {
          versionLabel: 'v0',
          runId: null,
          experimentName: 'exp04c_yolo26n_opentop_freeze10',
          status: 'BASELINE',
          datasetId: 'open-top-yolo-v1',
          valFingerprint: 'abc12345',
          valDatasetChanged: false,
          baseLabel: 'yolo26n.pt',
          createdAt: null,
          epochsCompleted: 60,
          map50: 0.9218,
          map5095: 0.6266,
          precision: 0.8394,
          recall: 0.9127,
          bestF1Conf: null,
          registeredModelId: 'exp04c_yolo26n_opentop_640',
          notes: null,
        },
      ],
      runs: [],
      advancedBaseOptions: [
        {
          id: 'auto',
          label: 'Automático — Baseline inicial (v0)',
          kind: 'auto',
          description: 'Contínuo',
        },
      ],
    };

    const apiMock = {
      getStatus: vi.fn().mockReturnValue(of(gpuStatus)),
      listCheckpoints: vi.fn().mockReturnValue(of([])),
      listRuns: vi.fn().mockReturnValue(of([])),
      getRun: vi.fn(),
      analyzeImage: vi.fn(),
      analyzeVideo: vi.fn(),
    };
    const queuedRunDetail = {
      runId: 'trn-1',
      versionNumber: 1,
      versionLabel: 'v1',
      experimentName: 'train_v1_open_top',
      status: 'COMPLETED' as const,
      datasetId: 'open-top-yolo-v1',
      valFingerprint: 'abc12345',
      baseId: 'checkpoint:exp04c_yolo26n_opentop_640',
      baseLabel: 'exp04c',
      architecture: 'yolo26n',
      epochsRequested: 30,
      epochsCompleted: 1,
      currentEpoch: 1,
      progressPercent: 100,
      imgsz: 640,
      batch: 8,
      freeze: 10,
      lr0: 0.0005,
      cosLr: true,
      patience: 15,
      bestMap50: 0.93,
      bestMap5095: 0.64,
      finalMap50: 0.93,
      finalMap5095: 0.64,
      finalPrecision: 0.88,
      finalRecall: 0.91,
      bestF1Conf: null,
      peakVramMib: 1400,
      wallTimeS: 12,
      bestWeightsPath: null,
      registeredModelId: null,
      notes: null,
      errorMessage: null,
      createdAt: '2026-11-24T21:00:00Z',
      startedAt: '2026-11-24T21:00:01Z',
      completedAt: '2026-11-24T21:00:13Z',
      epochHistory: [],
      logTail: [],
      valCalibrationCurve: [],
    };
    const trainingApiMock = {
      getOverview: vi.fn().mockReturnValue(of(trainingOverview)),
      getRun: vi.fn().mockReturnValue(of(queuedRunDetail)),
      startRun: vi.fn().mockReturnValue(of({ ...queuedRunDetail, status: 'QUEUED' as const })),
      stopRun: vi.fn(),
      registerCheckpoint: vi.fn(),
    };

    await TestBed.configureTestingModule({
      imports: [LabPageComponent],
      providers: [
        provideRouter([]),
        { provide: LabApiService, useValue: apiMock },
        { provide: TrainingApiService, useValue: trainingApiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(LabPageComponent);
    await fixture.whenStable();
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    (el.querySelector('[data-testid="lab-tab-training"]') as HTMLButtonElement).click();
    fixture.detectChanges();
    await fixture.whenStable();
    fixture.detectChanges();

    expect(el.querySelector('[data-testid="next-version-badge"]')?.textContent).toContain('v1');
    expect(el.querySelector('[data-testid="val-campaign-warning"]')?.textContent).toContain('1 campanha');
    expect(el.querySelector('[data-testid="recommended-base-label"]')?.textContent).toContain('exp04c');

    const startBtn = el.querySelector('[data-testid="start-training-btn"]') as HTMLButtonElement;
    expect(startBtn.textContent).toContain('Treinar versão v1');
    startBtn.click();
    expect(trainingApiMock.startRun).toHaveBeenCalledTimes(1);
  });
});
