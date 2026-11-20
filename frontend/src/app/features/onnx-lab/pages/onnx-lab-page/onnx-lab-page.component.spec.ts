import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { vi } from 'vitest';
import { NotificationService } from '../../../../core/services/notification.service';
import { ImageRunResult, RuntimeStatus, VideoRunStatus } from '../../models/onnx-lab.models';
import { OnnxLabApiService } from '../../services/onnx-lab-api.service';
import { OnnxLabPageComponent } from './onnx-lab-page.component';

describe('OnnxLabPageComponent', () => {
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

  it('renders live GPU provider status, optional ByteTrack notice, and zero-detection banner', async () => {
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
        originalUrl: '/api/v1/onnx-spike/runs/run-empty-1/files/original',
        annotatedUrl: '/api/v1/onnx-spike/runs/run-empty-1/files/annotated',
        originalDownloadUrl: '/api/v1/onnx-spike/runs/run-empty-1/files/original?download=true',
        annotatedDownloadUrl: '/api/v1/onnx-spike/runs/run-empty-1/files/annotated?download=true',
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
      imports: [OnnxLabPageComponent],
      providers: [
        { provide: OnnxLabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(OnnxLabPageComponent);
    await fixture.whenStable();

    const component = fixture.componentInstance;
    component.activeRun.set(emptyImageResult);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.textContent).toContain('NVIDIA GeForce RTX 4060 Ti');
    expect(el.querySelector('[data-testid="tracking-limitation-notice"]')?.textContent).toContain(
      'ByteTrack'
    );
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
        originalUrl: '/api/v1/onnx-spike/runs/run-vid-py-1/files/original',
        annotatedUrl: '/api/v1/onnx-spike/runs/run-vid-py-1/files/annotated',
        originalDownloadUrl: '/api/v1/onnx-spike/runs/run-vid-py-1/files/original?download=true',
        annotatedDownloadUrl: '/api/v1/onnx-spike/runs/run-vid-py-1/files/annotated?download=true',
        originalFilename: 'original.ogv',
        annotatedFilename: 'annotated.mp4',
        annotatedContentType: 'video/mp4',
      },
      frameSummaries: [
        {
          frameIndex: 0,
          timestampMs: 0,
          detectionCount: 1,
          frameLatencyMs: 18.0,
          detections: [
            {
              classId: 0,
              className: 'open_top_freight_wagon',
              confidence: 0.85,
              trackId: 1,
              x1: 100,
              y1: 200,
              x2: 500,
              y2: 600,
              widthPx: 400,
              heightPx: 400,
              normX: 0.05,
              normY: 0.18,
              normWidth: 0.2,
              normHeight: 0.37,
            },
          ],
        },
      ],
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
      imports: [OnnxLabPageComponent],
      providers: [
        { provide: OnnxLabApiService, useValue: apiMock },
        {
          provide: NotificationService,
          useValue: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
        },
      ],
    }).compileComponents();

    const fixture = TestBed.createComponent(OnnxLabPageComponent);
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
});
