export type { ModelCheckpointSummary, RuntimeOption, RuntimeStatus } from '../../../core/models/runtime.models';
import type { ModelCheckpointSummary } from '../../../core/models/runtime.models';

export type ModelCheckpoint = ModelCheckpointSummary;

export interface DetectionBox {
  classId: number;
  className: string;
  confidence: number;
  trackId: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  widthPx: number;
  heightPx: number;
  normX: number;
  normY: number;
  normWidth: number;
  normHeight: number;
}

export interface InferenceTimingMetrics {
  preprocessMs: number;
  inferenceMs: number;
  postprocessMs: number;
  decodeOrEncodeMs: number;
  totalMs: number;
  avgFrameMs: number;
  p50FrameMs: number;
  p95FrameMs: number;
  processingFps: number;
  inferenceOnlyFps: number;
}

export interface ModelConfigUsed {
  modelName: string;
  sourceExperiment: string;
  modelSha256: string;
  inputSize: number;
  confThreshold: number;
  iouThreshold: number;
  sampleFps: number | null;
  deviceType: string;
  activeExecutionProvider: string;
  trackingIntegrated: boolean;
  trackingStatus: string;
  exploratoryNotice: string;
  runtimeId?: string | null;
  modelId?: string | null;
  fallbackOccurred?: boolean;
  outputResolution?: string | null;
  trackerAlgorithm?: string | null;
}

export interface RunFileLinks {
  originalUrl: string;
  annotatedUrl: string | null;
  originalDownloadUrl: string;
  annotatedDownloadUrl: string | null;
  originalFilename: string;
  annotatedFilename: string | null;
  annotatedContentType: string | null;
}

export interface ImageRunResult {
  runId: string;
  mediaType: 'IMAGE';
  status: string;
  createdAt: string;
  originalFilename: string;
  imageWidth: number;
  imageHeight: number;
  detectionCount: number;
  noDetections: boolean;
  emptyMessage: string | null;
  detections: DetectionBox[];
  metrics: InferenceTimingMetrics;
  modelConfig: ModelConfigUsed;
  files: RunFileLinks;
}

export interface VideoFrameSummary {
  frameIndex: number;
  timestampMs: number;
  detectionCount: number;
  frameLatencyMs: number;
  detections: DetectionBox[];
}

export interface VideoRunStatus {
  runId: string;
  mediaType: 'VIDEO';
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  originalFilename: string;
  videoWidth: number;
  videoHeight: number;
  sourceFps: number;
  sourceCodec: string;
  sourceDurationMs: number;
  outputCodec: string | null;
  outputContainer: string | null;
  browserPlayable: boolean;
  estimatedTotalFrames: number;
  processedFrames: number;
  progressPercent: number;
  framesWithDetections: number;
  totalDetections: number;
  noDetections: boolean;
  truncatedByFrameLimit: boolean;
  emptyMessage: string | null;
  errorMessage: string | null;
  metrics: InferenceTimingMetrics | null;
  modelConfig: ModelConfigUsed;
  files: RunFileLinks;
  frameSummaries: VideoFrameSummary[];
  frameSummariesTruncated?: boolean;
  totalFrameSummaries?: number | null;
  outputResolution?: string | null;
  outputWidth?: number | null;
  outputHeight?: number | null;
  outputEncoder?: string | null;
  outputCrf?: number | null;
  uniqueTrackCount?: number | null;
}

export type LabRunResult = ImageRunResult | VideoRunStatus;

export interface RunListItem {
  runId: string;
  mediaType: 'IMAGE' | 'VIDEO';
  status: string;
  originalFilename: string;
  createdAt: string;
  totalDetections: number;
  noDetections: boolean;
  processingFps: number | null;
  avgFrameMs: number | null;
  totalMs: number | null;
  deviceType: string;
  activeExecutionProvider: string;
  modelName: string;
  modelId?: string | null;
  runtimeId?: string | null;
  trackingIntegrated?: boolean;
  uniqueTrackCount?: number | null;
}

export interface RunComparisonItem {
  runId: string;
  mediaType: string;
  status: string;
  originalFilename: string;
  modelName: string;
  modelId?: string | null;
  modelSha256?: string | null;
  sourceExperiment: string;
  inputSize?: number | null;
  runtimeId?: string | null;
  deviceType: string;
  confThreshold?: number | null;
  iouThreshold?: number | null;
  sampleFps?: number | null;
  sourceResolution?: string | null;
  outputResolution?: string | null;
  trackingIntegrated?: boolean;
  totalDetections: number;
  uniqueTrackCount?: number | null;
  processingFps?: number | null;
  inferenceOnlyFps?: number | null;
  avgFrameMs?: number | null;
  p95FrameMs?: number | null;
}

export interface FrameComparisonDelta {
  frameIndex: number;
  timestampMs: number;
  detectionsA: number;
  detectionsB: number;
  delta: number;
}

export interface RunComparison {
  runA: RunComparisonItem;
  runB: RunComparisonItem;
  comparable: boolean;
  comparabilityWarnings: string[];
  deltaDetections: number;
  deltaProcessingFps?: number | null;
  deltaAvgFrameMs?: number | null;
  frameDeltas: FrameComparisonDelta[];
}

export interface CurateFrameRequest {
  frameIndex?: number | null;
  reason: string;
  notes?: string | null;
  targetCampaign?: string | null;
}

export interface CurationExport {
  runId: string;
  frameIndex: number;
  imagePath: string;
  labelPath: string;
  metadataPath: string;
  sourceSha256?: string;
  modelSha256?: string | null;
  reservedBenchmarkWarning?: boolean;
  detectionCount: number;
  reason: string;
  humanReviewed: boolean;
  autoTrainEligible: boolean;
  createdAt: string;
}
