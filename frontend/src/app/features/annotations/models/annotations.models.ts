export type AnnotationItemStatus =
  | 'pending'
  | 'pre_annotated'
  | 'in_review'
  | 'reviewed'
  | 'approved'
  | 'rejected';

export type BoxOrigin = 'manual' | 'ai_suggestion' | 'ai_accepted' | 'ai_edited';

export interface AnnotationBox {
  id: string;
  classId: number;
  className: string;
  x: number;
  y: number;
  width: number;
  height: number;
  origin: BoxOrigin;
  confidence?: number | null;
  modelId?: string | null;
  modelSha256?: string | null;
}

export interface AnnotationProjectSummary {
  id: string;
  name: string;
  campaignId: string;
  sourceOrigin: string;
  description: string | null;
  classes: string[];
  itemCount: number;
  videoCount: number;
  statusCounts: Record<string, number>;
  reviewedCount: number;
  approvedCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface AnnotationVideo {
  id: string;
  filename: string;
  sourceOrigin: string;
  campaignId: string;
  sha256: string;
  widthPx: number;
  heightPx: number;
  fps: number;
  durationMs: number;
  codec: string;
  estimatedFrames: number;
  reservedBenchmarkWarning: boolean;
  importedAt: string;
}

export interface AnnotationItemSummary {
  id: string;
  projectId: string;
  filename: string;
  sourceOrigin: string;
  campaignId: string;
  sourceSha256: string;
  widthPx: number;
  heightPx: number;
  status: AnnotationItemStatus;
  humanReviewed: boolean;
  noObjectsConfirmed: boolean;
  boxCount: number;
  aiSuggestionCount: number;
  reservedBenchmarkWarning: boolean;
  sourceVideoId: string | null;
  sourceVideoFilename: string | null;
  frameIndex: number | null;
  timestampMs: number | null;
  imageUrl: string;
  thumbnailUrl: string;
  createdAt: string;
  updatedAt: string;
  reviewedAt: string | null;
  approvedAt: string | null;
}

export interface AnnotationItemDetail extends AnnotationItemSummary {
  boxes: AnnotationBox[];
  reviewNotes: string | null;
  falsePositivesRemoved: number;
  falseNegativesAdded: number;
  preannotationModelId: string | null;
  preannotationModelSha256: string | null;
  preannotationConf: number | null;
  preannotationIou: number | null;
}

export interface AnnotationProjectDetail extends AnnotationProjectSummary {
  items: AnnotationItemSummary[];
  videos: AnnotationVideo[];
}

export interface CreateAnnotationProjectRequest {
  name: string;
  campaignId: string;
  sourceOrigin: string;
  description?: string | null;
  classes?: string[];
}

export interface UpdateItemAnnotationsRequest {
  boxes: AnnotationBox[];
  noObjectsConfirmed: boolean;
  status?: AnnotationItemStatus | null;
  reviewNotes?: string | null;
  falsePositivesRemoved?: number;
  falseNegativesAdded?: number;
}

export interface PreviewFrameCandidate {
  frameIndex: number;
  timestampMs: number;
  diffScore: number;
  previewUrl: string;
}

export interface PreviewFramesResponse {
  videoId: string;
  candidates: PreviewFrameCandidate[];
  scannedFrames: number;
  skippedSimilar: number;
}

export interface PreannotateBatchResponse {
  projectId: string;
  processedItems: number;
  skippedReviewed: number;
  totalSuggestions: number;
  modelId: string;
  runtimeId: string;
}

export interface ImportCurationResponse {
  importedCount: number;
  skippedDuplicates: number;
  items: AnnotationItemSummary[];
}
