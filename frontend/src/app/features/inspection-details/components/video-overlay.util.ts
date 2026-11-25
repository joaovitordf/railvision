export interface ContainedVideoRect {
  offsetX: number;
  offsetY: number;
  drawWidth: number;
  drawHeight: number;
}

export function computeContainedVideoRect(
  containerWidth: number,
  containerHeight: number,
  videoWidth: number | null | undefined,
  videoHeight: number | null | undefined
): ContainedVideoRect {
  if (containerWidth <= 0 || containerHeight <= 0) {
    return { offsetX: 0, offsetY: 0, drawWidth: 0, drawHeight: 0 };
  }
  if (!videoWidth || !videoHeight || videoWidth <= 0 || videoHeight <= 0) {
    return { offsetX: 0, offsetY: 0, drawWidth: containerWidth, drawHeight: containerHeight };
  }
  const containerAspect = containerWidth / containerHeight;
  const videoAspect = videoWidth / videoHeight;
  if (videoAspect > containerAspect) {
    const drawWidth = containerWidth;
    const drawHeight = containerWidth / videoAspect;
    return {
      offsetX: 0,
      offsetY: (containerHeight - drawHeight) / 2,
      drawWidth,
      drawHeight,
    };
  }
  const drawHeight = containerHeight;
  const drawWidth = containerHeight * videoAspect;
  return {
    offsetX: (containerWidth - drawWidth) / 2,
    offsetY: 0,
    drawWidth,
    drawHeight,
  };
}
