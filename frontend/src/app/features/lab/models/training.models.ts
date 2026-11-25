export type TrainingRunStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface TrainingDatasetOption {
  id: string;
  name: string;
  dataYamlPath: string;
  trainCount: number;
  valCount: number;
  valCampaigns: string[];
  valFingerprint: string;
  hasFrozenTest: boolean;
  warning: string | null;
}

export interface TrainingBaseOption {
  id: string;
  label: string;
  kind: 'auto' | 'run' | 'checkpoint' | 'pretrained';
  description: string;
}

export interface TrainingEpochPoint {
  epoch: number;
  timeSec: number;
  trainBoxLoss: number;
  trainClsLoss: number;
  trainL1Loss: number;
  valBoxLoss: number;
  valClsLoss: number;
  valL1Loss: number;
  precision: number;
  recall: number;
  map50: number;
  map5095: number;
  lr: number | null;
}

export interface TrainingLineagePoint {
  versionLabel: string;
  runId: string | null;
  experimentName: string;
  status: string;
  datasetId: string;
  valFingerprint: string | null;
  valDatasetChanged: boolean;
  baseLabel: string;
  createdAt: string | null;
  epochsCompleted: number;
  map50: number | null;
  map5095: number | null;
  precision: number | null;
  recall: number | null;
  bestF1Conf: number | null;
  registeredModelId: string | null;
  notes: string | null;
}

export interface TrainingRunSummary {
  runId: string;
  versionNumber: number;
  versionLabel: string;
  experimentName: string;
  status: TrainingRunStatus;
  datasetId: string;
  valFingerprint: string;
  baseId: string;
  baseLabel: string;
  architecture: string;
  epochsRequested: number;
  epochsCompleted: number;
  currentEpoch: number;
  progressPercent: number;
  imgsz: number;
  batch: number;
  freeze: number;
  lr0: number;
  cosLr: boolean;
  patience: number;
  bestMap50: number | null;
  bestMap5095: number | null;
  finalMap50: number | null;
  finalMap5095: number | null;
  finalPrecision: number | null;
  finalRecall: number | null;
  bestF1Conf: number | null;
  peakVramMib: number | null;
  wallTimeS: number | null;
  bestWeightsPath: string | null;
  registeredModelId: string | null;
  notes: string | null;
  errorMessage: string | null;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
}

export interface CalibrationCurvePoint {
  conf: number;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface TrainingRunDetail extends TrainingRunSummary {
  epochHistory: TrainingEpochPoint[];
  logTail: string[];
  valCalibrationCurve: CalibrationCurvePoint[];
}

export interface TrainingBaselineMetrics {
  modelId: string;
  versionLabel: string;
  sourceExperiment: string;
  map50: number | null;
  map5095: number | null;
  precision: number | null;
  recall: number | null;
}

export interface TrainingOverview {
  lineageName: string;
  baselineModelId: string;
  baselineMetrics: TrainingBaselineMetrics;
  nextVersionNumber: number;
  nextVersionLabel: string;
  recommendedBaseId: string;
  recommendedBaseLabel: string;
  activeRunId: string | null;
  gpuAvailable: boolean;
  gpuName: string | null;
  datasets: TrainingDatasetOption[];
  lineagePoints: TrainingLineagePoint[];
  runs: TrainingRunSummary[];
  advancedBaseOptions: TrainingBaseOption[];
}

export interface StartTrainingRequest {
  datasetId: string;
  baseOverride?: string | null;
  epochs: number;
  patience: number;
  imgsz: number;
  batch: number;
  freeze: number;
  lr0: number;
  cosLr: boolean;
  notes?: string | null;
}

export interface RegisterTrainingCheckpointRequest {
  modelId?: string | null;
  activate?: boolean;
}
