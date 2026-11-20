export type InspectionStatus = 'UPLOADED' | 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type JobStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';

export type LoadingLevel = 'empty' | 'low' | 'medium' | 'high' | 'very_high';

export interface ProblemDetail {
  type?: string;
  title?: string;
  status?: number;
  detail?: string;
  instance?: string;
  errorCode?: string;
}

export interface InspectionSummary {
  id: string;
  title: string;
  notes: string | null;
  status: InspectionStatus;
  originalFilename: string;
  contentType: string;
  fileSizeBytes: number;
  sha256Checksum: string;
  latestJobId: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface ProcessingJob {
  id: string;
  inspectionId: string;
  status: JobStatus;
  progressPercent: number | null;
  engineType: string;
  isDemoSimulation: boolean;
  errorCode: string | null;
  errorMessage: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  createdAt: string;
}

export interface InspectionDetail extends InspectionSummary {
  durationMs: number | null;
  widthPx: number | null;
  heightPx: number | null;
  fps: number | null;
  codec: string | null;
  wagonCount: number;
  reviewRequiredCount: number;
  latestJob: ProcessingJob | null;
}

export interface WagonIdentity {
  trackId: string;
  sequenceIndex: number;
  physicalId: string | null;
}

export interface MaterialEstimate {
  materialClass: string;
  confidence: number | null;
  voteDistribution: Record<string, number>;
  rejectedAsUnknown: boolean;
}

export interface LoadingEstimate {
  level: LoadingLevel;
  levelObservationSpread: string[];
  estimatedFillPercentage: number | null;
  confidence: number | null;
  measurementType: string;
  physicalVolumeVerified: boolean;
}

export interface ConsolidationInfo {
  observationsCount: number;
  disagreements: string[];
  method: string | null;
}

export interface WagonResult {
  id: string;
  inspectionId: string;
  jobId: string;
  wagonIdentity: WagonIdentity;
  material: MaterialEstimate | null;
  loadingEstimate: LoadingEstimate;
  modelVersions: Record<string, string>;
  consolidation: ConsolidationInfo;
  qualityFlags: string[];
  reviewRequired: boolean;
  reviewReasons: string[];
  isExample: boolean;
  createdAt: string;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  coordinateSystem: string;
}

export interface WagonObservation {
  id: string;
  wagonResultId: string;
  frameIndex: number;
  frameTimestampMs: number;
  boundingBox: BoundingBox;
  detectedClass: string;
  detectionConfidence: number;
  materialClass: string | null;
  materialConfidence: number | null;
  loadingLevel: string | null;
  loadingConfidence: number | null;
  isExample: boolean;
}

export interface ModelVersion {
  id: string;
  task: string;
  name: string;
  version: string;
  runtime: string;
  weightsSha256: string | null;
  artifactRef: string | null;
  isDemoPlaceholder: boolean;
  createdAt: string;
}
