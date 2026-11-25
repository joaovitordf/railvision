export interface RuntimeOption {
  id: string;
  label: string;
  deviceType: string;
  available: boolean;
  provider: string;
  diagnostic: string;
}

export interface RuntimeStatus {
  available: boolean;
  unavailableReason: string | null;
  modelName: string;
  sourceExperiment: string;
  modelDisplayPath: string;
  modelSha256: string;
  inputSize: number;
  inputShape: number[];
  outputShape: number[];
  classNames: Record<string, string>;
  requestedExecutionProvider: string;
  activeExecutionProvider: string;
  deviceType: string;
  cudaAvailable: boolean;
  fallbackOccurred: boolean;
  availableProviders: string[];
  deviceDiagnostic: string;
  trackingIntegrated: boolean;
  trackingStatus: string;
  trackingNotice: string;
  limitationsNotice: string;
  defaultConfThreshold: number;
  defaultIouThreshold: number;
  maxVideoFramesPerUiRun: number;
  supportedImageExtensions: string[];
  supportedVideoExtensions: string[];
  backendEngine?: string;
  defaultRuntime?: string;
  availableRuntimes?: RuntimeOption[];
  defaultOutputResolution?: string;
  availableOutputResolutions?: string[];
  videoEncoder?: string;
  videoEncoderCrf?: number;
  trackingAlgorithms?: string[];
  activeModelId?: string | null;
}

export interface ModelCheckpointSummary {
  id: string;
  modelName: string;
  sourceExperiment: string;
  architecture: string;
  status: string;
  inputSize: number;
  classNames: Record<string, string>;
  pytorchWeightsPath: string | null;
  pytorchWeightsSha256: string | null;
  onnxModelPath: string | null;
  onnxMetaPath: string | null;
  onnxModelSha256: string | null;
  limitationsNotice: string;
  valMetrics: Record<string, number>;
  testMetrics: Record<string, number>;
}
