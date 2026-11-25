import { clampNormalizedBox } from './annotation-canvas.component';

describe('clampNormalizedBox', () => {
  it('clamps coordinates inside [0, 1] and normalizes reverse drags', () => {
    const reversed = clampNormalizedBox(0.6, 0.7, -0.35, -0.2);
    expect(reversed).toEqual({
      x: 0.25,
      y: 0.5,
      width: 0.35,
      height: 0.2,
    });

    const outOfBounds = clampNormalizedBox(-0.2, 0.85, 0.5, 0.4);
    expect(outOfBounds).toEqual({
      x: 0,
      y: 0.85,
      width: 0.3,
      height: 0.15,
    });
  });
});
