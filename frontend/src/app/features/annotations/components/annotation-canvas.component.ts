import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  ViewChild,
  effect,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import { AnnotationBox } from '../models/annotations.models';

export type EditorTool = 'select' | 'draw' | 'pan';

type HandleKind = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

interface InteractionState {
  mode: 'none' | 'pan' | 'draw' | 'move' | 'resize';
  startClientX: number;
  startClientY: number;
  origPanX: number;
  origPanY: number;
  drawStartNormX: number;
  drawStartNormY: number;
  drawCurNormX: number;
  drawCurNormY: number;
  boxId: string | null;
  origBox: AnnotationBox | null;
  handle: HandleKind | null;
}

const MIN_BOX_NORM = 0.008;

export function clampNormalizedBox(x: number, y: number, w: number, h: number): {
  x: number;
  y: number;
  width: number;
  height: number;
} {
  const x1 = Math.max(0, Math.min(1, x));
  const y1 = Math.max(0, Math.min(1, y));
  const x2 = Math.max(0, Math.min(1, x + w));
  const y2 = Math.max(0, Math.min(1, y + h));
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const width = Math.max(0, Math.min(1 - left, Math.abs(x2 - x1)));
  const height = Math.max(0, Math.min(1 - top, Math.abs(y2 - y1)));
  return {
    x: Number(left.toFixed(6)),
    y: Number(top.toFixed(6)),
    width: Number(width.toFixed(6)),
    height: Number(height.toFixed(6)),
  };
}

@Component({
  selector: 'rv-annotation-canvas',
  standalone: true,
  template: `
    <div #stage class="rv-ann-canvas-stage">
      <canvas
        #canvas
        [style.cursor]="canvasCursor()"
        (pointerdown)="onPointerDown($event)"
        (pointermove)="onPointerMove($event)"
        (pointerup)="onPointerUp($event)"
        (pointerleave)="onPointerUp($event)"
        (wheel)="onWheel($event)"
      ></canvas>

      @if (noObjectsConfirmed()) {
        <div class="rv-ann-negative-overlay" data-testid="negative-confirmed-overlay">
          Imagem confirmada sem vagões graneleiros abertos (hard negative)
        </div>
      }

      @if (!imageLoaded() && !imageFailed()) {
        <div class="rv-ann-loading-overlay">Carregando imagem...</div>
      }

      @if (imageFailed()) {
        <div class="rv-ann-loading-overlay">Não foi possível carregar esta imagem.</div>
      }
    </div>
  `,
  styles: [
    `
      .rv-ann-canvas-stage {
        position: relative;
        width: 100%;
        height: 100%;
        background: #0b111c;
        overflow: hidden;
        user-select: none;
        touch-action: none;
      }

      .rv-ann-canvas-stage canvas {
        width: 100%;
        height: 100%;
        display: block;
      }

      .rv-ann-negative-overlay {
        position: absolute;
        top: 12px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(217, 119, 6, 0.92);
        color: #ffffff;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        pointer-events: none;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
      }

      .rv-ann-loading-overlay {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        color: #94a3b8;
        font-size: 0.92rem;
        pointer-events: none;
      }
    `,
  ],
})
export class AnnotationCanvasComponent implements AfterViewInit {
  private readonly destroyRef = inject(DestroyRef);

  readonly imageUrl = input<string | null>(null);
  readonly boxes = input<AnnotationBox[]>([]);
  readonly selectedBoxId = input<string | null>(null);
  readonly tool = input<EditorTool>('select');
  readonly activeClassName = input<string>('open_top_freight_wagon');
  readonly noObjectsConfirmed = input<boolean>(false);

  readonly boxSelected = output<string | null>();
  readonly boxCreated = output<Omit<AnnotationBox, 'id'>>();
  readonly boxUpdated = output<AnnotationBox>();
  readonly zoomChanged = output<number>();

  @ViewChild('stage') private stageRef?: ElementRef<HTMLDivElement>;
  @ViewChild('canvas') private canvasRef?: ElementRef<HTMLCanvasElement>;

  protected readonly imageLoaded = signal<boolean>(false);
  protected readonly imageFailed = signal<boolean>(false);
  protected readonly zoom = signal<number>(1);
  protected readonly panX = signal<number>(0);
  protected readonly panY = signal<number>(0);

  private imgEl: HTMLImageElement | null = null;
  private resizeObserver?: ResizeObserver;
  private interaction: InteractionState = this.emptyInteraction();

  constructor() {
    effect(() => {
      const url = this.imageUrl();
      this.loadImage(url);
    });

    effect(() => {
      this.boxes();
      this.selectedBoxId();
      this.zoom();
      this.panX();
      this.panY();
      queueMicrotask(() => this.redraw());
    });
  }

  ngAfterViewInit(): void {
    const stage = this.stageRef?.nativeElement;
    if (stage && typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.fitIfDefault());
      this.resizeObserver.observe(stage);
      this.destroyRef.onDestroy(() => this.resizeObserver?.disconnect());
    }
  }

  zoomIn(): void {
    this.zoomAtCenter(1.25);
  }

  zoomOut(): void {
    this.zoomAtCenter(0.8);
  }

  fitScreen(): void {
    this.zoom.set(1);
    this.panX.set(0);
    this.panY.set(0);
    this.zoomChanged.emit(1);
    this.redraw();
  }

  protected canvasCursor(): string {
    if (this.interaction.mode === 'pan' || this.tool() === 'pan') {
      return 'grab';
    }
    if (this.tool() === 'draw') {
      return 'crosshair';
    }
    return 'default';
  }

  protected onPointerDown(event: PointerEvent): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas || !this.imgEl || !this.imageLoaded()) {
      return;
    }
    canvas.setPointerCapture?.(event.pointerId);
    const { x: cx, y: cy } = this.localPoint(event, canvas);

    if (event.button === 1 || this.tool() === 'pan' || (event.button === 0 && event.shiftKey)) {
      this.interaction = {
        ...this.emptyInteraction(),
        mode: 'pan',
        startClientX: cx,
        startClientY: cy,
        origPanX: this.panX(),
        origPanY: this.panY(),
      };
      return;
    }

    const norm = this.canvasToNorm(cx, cy, canvas);
    if (this.tool() === 'draw') {
      const clampedX = Math.max(0, Math.min(1, norm.x));
      const clampedY = Math.max(0, Math.min(1, norm.y));
      this.interaction = {
        ...this.emptyInteraction(),
        mode: 'draw',
        drawStartNormX: clampedX,
        drawStartNormY: clampedY,
        drawCurNormX: clampedX,
        drawCurNormY: clampedY,
      };
      this.boxSelected.emit(null);
      return;
    }

    // Select mode: check resize handles on currently selected box first
    const selectedId = this.selectedBoxId();
    const currentBoxes = this.boxes();
    if (selectedId) {
      const selBox = currentBoxes.find((b) => b.id === selectedId);
      if (selBox) {
        const hitHandle = this.hitTestHandle(cx, cy, selBox, canvas);
        if (hitHandle) {
          this.interaction = {
            ...this.emptyInteraction(),
            mode: 'resize',
            startClientX: cx,
            startClientY: cy,
            drawStartNormX: norm.x,
            drawStartNormY: norm.y,
            boxId: selBox.id,
            origBox: { ...selBox },
            handle: hitHandle,
          };
          return;
        }
      }
    }

    // Hit test any box top-to-bottom (smallest or topmost)
    const hitBox = [...currentBoxes]
      .reverse()
      .find(
        (b) =>
          norm.x >= b.x &&
          norm.x <= b.x + b.width &&
          norm.y >= b.y &&
          norm.y <= b.y + b.height
      );

    if (hitBox) {
      this.boxSelected.emit(hitBox.id);
      this.interaction = {
        ...this.emptyInteraction(),
        mode: 'move',
        startClientX: cx,
        startClientY: cy,
        drawStartNormX: norm.x,
        drawStartNormY: norm.y,
        boxId: hitBox.id,
        origBox: { ...hitBox },
      };
      return;
    }

    this.boxSelected.emit(null);
  }

  protected onPointerMove(event: PointerEvent): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas || this.interaction.mode === 'none') {
      return;
    }
    const { x: cx, y: cy } = this.localPoint(event, canvas);

    if (this.interaction.mode === 'pan') {
      this.panX.set(this.interaction.origPanX + (cx - this.interaction.startClientX));
      this.panY.set(this.interaction.origPanY + (cy - this.interaction.startClientY));
      return;
    }

    const norm = this.canvasToNorm(cx, cy, canvas);
    if (this.interaction.mode === 'draw') {
      this.interaction.drawCurNormX = Math.max(0, Math.min(1, norm.x));
      this.interaction.drawCurNormY = Math.max(0, Math.min(1, norm.y));
      this.redraw();
      return;
    }

    if (this.interaction.mode === 'move' && this.interaction.origBox) {
      const dx = norm.x - this.interaction.drawStartNormX;
      const dy = norm.y - this.interaction.drawStartNormY;
      const ob = this.interaction.origBox;
      const nx = Math.max(0, Math.min(1 - ob.width, ob.x + dx));
      const ny = Math.max(0, Math.min(1 - ob.height, ob.y + dy));
      const nextOrigin = ob.origin === 'ai_suggestion' ? 'ai_edited' : ob.origin;
      this.boxUpdated.emit({
        ...ob,
        x: Number(nx.toFixed(6)),
        y: Number(ny.toFixed(6)),
        origin: nextOrigin,
      });
      return;
    }

    if (this.interaction.mode === 'resize' && this.interaction.origBox && this.interaction.handle) {
      const ob = this.interaction.origBox;
      let left = ob.x;
      let top = ob.y;
      let right = ob.x + ob.width;
      let bottom = ob.y + ob.height;
      const h = this.interaction.handle;
      const nx = Math.max(0, Math.min(1, norm.x));
      const ny = Math.max(0, Math.min(1, norm.y));

      if (h.includes('w')) left = Math.min(right - MIN_BOX_NORM, nx);
      if (h.includes('e')) right = Math.max(left + MIN_BOX_NORM, nx);
      if (h.includes('n')) top = Math.min(bottom - MIN_BOX_NORM, ny);
      if (h.includes('s')) bottom = Math.max(top + MIN_BOX_NORM, ny);

      const clamped = clampNormalizedBox(left, top, right - left, bottom - top);
      const nextOrigin = ob.origin === 'ai_suggestion' ? 'ai_edited' : ob.origin;
      this.boxUpdated.emit({
        ...ob,
        ...clamped,
        origin: nextOrigin,
      });
    }
  }

  protected onPointerUp(event: PointerEvent): void {
    const canvas = this.canvasRef?.nativeElement;
    if (canvas?.hasPointerCapture?.(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }

    if (this.interaction.mode === 'draw') {
      const clamped = clampNormalizedBox(
        this.interaction.drawStartNormX,
        this.interaction.drawStartNormY,
        this.interaction.drawCurNormX - this.interaction.drawStartNormX,
        this.interaction.drawCurNormY - this.interaction.drawStartNormY
      );
      if (clamped.width >= MIN_BOX_NORM && clamped.height >= MIN_BOX_NORM) {
        this.boxCreated.emit({
          classId: 0,
          className: this.activeClassName(),
          x: clamped.x,
          y: clamped.y,
          width: clamped.width,
          height: clamped.height,
          origin: 'manual',
          confidence: null,
        });
      }
    }

    this.interaction = this.emptyInteraction();
    this.redraw();
  }

  protected onWheel(event: WheelEvent): void {
    event.preventDefault();
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) {
      return;
    }
    const factor = event.deltaY < 0 ? 1.15 : 0.87;
    const prevZoom = this.zoom();
    const nextZoom = Math.max(0.5, Math.min(8.0, Number((prevZoom * factor).toFixed(3))));
    if (nextZoom === prevZoom) {
      return;
    }

    const { x: cx, y: cy } = this.localPoint(event, canvas);
    const ratio = nextZoom / prevZoom;
    const rect = this.computeDrawRect(canvas.clientWidth, canvas.clientHeight);
    const centerShiftX = (cx - rect.stageCenterX - this.panX()) * (ratio - 1);
    const centerShiftY = (cy - rect.stageCenterY - this.panY()) * (ratio - 1);

    this.panX.update((px) => px - centerShiftX);
    this.panY.update((py) => py - centerShiftY);
    this.zoom.set(nextZoom);
    this.zoomChanged.emit(nextZoom);
  }

  private zoomAtCenter(factor: number): void {
    const nextZoom = Math.max(0.5, Math.min(8.0, Number((this.zoom() * factor).toFixed(3))));
    this.zoom.set(nextZoom);
    this.zoomChanged.emit(nextZoom);
  }

  private loadImage(url: string | null): void {
    this.imageLoaded.set(false);
    this.imageFailed.set(false);
    this.imgEl = null;
    this.fitScreen();
    if (!url || typeof Image === 'undefined') {
      return;
    }
    const img = new Image();
    img.onload = () => {
      this.imgEl = img;
      this.imageLoaded.set(true);
      this.imageFailed.set(false);
      this.redraw();
    };
    img.onerror = () => {
      this.imageLoaded.set(false);
      this.imageFailed.set(true);
    };
    img.src = url;
  }

  private fitIfDefault(): void {
    this.redraw();
  }

  private computeDrawRect(viewW: number, viewH: number): {
    x: number;
    y: number;
    w: number;
    h: number;
    stageCenterX: number;
    stageCenterY: number;
  } {
    const imgW = this.imgEl?.naturalWidth || 1;
    const imgH = this.imgEl?.naturalHeight || 1;
    const baseScale = Math.min((viewW - 24) / imgW, (viewH - 24) / imgH);
    const scale = Math.max(0.05, baseScale) * this.zoom();
    const w = imgW * scale;
    const h = imgH * scale;
    const stageCenterX = viewW / 2;
    const stageCenterY = viewH / 2;
    return {
      x: stageCenterX - w / 2 + this.panX(),
      y: stageCenterY - h / 2 + this.panY(),
      w,
      h,
      stageCenterX,
      stageCenterY,
    };
  }

  private localPoint(event: MouseEvent, canvas: HTMLCanvasElement): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
  }

  private canvasToNorm(cx: number, cy: number, canvas: HTMLCanvasElement): { x: number; y: number } {
    const dr = this.computeDrawRect(canvas.clientWidth, canvas.clientHeight);
    return {
      x: (cx - dr.x) / dr.w,
      y: (cy - dr.y) / dr.h,
    };
  }

  private hitTestHandle(cx: number, cy: number, box: AnnotationBox, canvas: HTMLCanvasElement): HandleKind | null {
    const dr = this.computeDrawRect(canvas.clientWidth, canvas.clientHeight);
    const x1 = dr.x + box.x * dr.w;
    const y1 = dr.y + box.y * dr.h;
    const x2 = x1 + box.width * dr.w;
    const y2 = y1 + box.height * dr.h;
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;

    const handles: [HandleKind, number, number][] = [
      ['nw', x1, y1],
      ['n', mx, y1],
      ['ne', x2, y1],
      ['e', x2, my],
      ['se', x2, y2],
      ['s', mx, y2],
      ['sw', x1, y2],
      ['w', x1, my],
    ];
    for (const [kind, hx, hy] of handles) {
      if (Math.abs(cx - hx) <= 8 && Math.abs(cy - hy) <= 8) {
        return kind;
      }
    }
    return null;
  }

  redraw(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) {
      return;
    }
    const viewW = canvas.clientWidth || 800;
    const viewH = canvas.clientHeight || 500;
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    if (canvas.width !== Math.round(viewW * dpr) || canvas.height !== Math.round(viewH * dpr)) {
      canvas.width = Math.round(viewW * dpr);
      canvas.height = Math.round(viewH * dpr);
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, viewW, viewH);

    if (!this.imgEl || !this.imageLoaded()) {
      return;
    }

    const dr = this.computeDrawRect(viewW, viewH);
    ctx.drawImage(this.imgEl, dr.x, dr.y, dr.w, dr.h);

    const selId = this.selectedBoxId();
    for (const box of this.boxes()) {
      const bx = dr.x + box.x * dr.w;
      const by = dr.y + box.y * dr.h;
      const bw = box.width * dr.w;
      const bh = box.height * dr.h;
      const isSelected = box.id === selId;
      const isAi = box.origin === 'ai_suggestion';

      ctx.save();
      ctx.strokeStyle = isSelected ? '#3b82f6' : isAi ? '#f59e0b' : '#22c55e';
      ctx.lineWidth = isSelected ? 2.5 : 2;
      if (isAi && !isSelected) {
        ctx.setLineDash([6, 4]);
      }
      ctx.strokeRect(bx, by, bw, bh);

      ctx.fillStyle = isSelected
        ? 'rgba(59, 130, 246, 0.14)'
        : isAi
        ? 'rgba(245, 158, 11, 0.10)'
        : 'rgba(34, 197, 94, 0.10)';
      ctx.fillRect(bx, by, bw, bh);
      ctx.restore();

      const confSuffix = box.confidence != null ? ` ${Math.round(box.confidence * 100)}%` : '';
      const prefix = isAi ? 'IA · ' : '';
      const tag = `${prefix}${box.className}${confSuffix}`;
      ctx.font = '600 11px sans-serif';
      const tw = ctx.measureText(tag).width;
      ctx.fillStyle = isSelected ? '#1d4ed8' : isAi ? '#b45309' : '#15803d';
      ctx.fillRect(bx, Math.max(0, by - 19), tw + 10, 18);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(tag, bx + 5, Math.max(13, by - 6));

      if (isSelected) {
        const mx = bx + bw / 2;
        const my = by + bh / 2;
        const pts: [number, number][] = [
          [bx, by],
          [mx, by],
          [bx + bw, by],
          [bx + bw, my],
          [bx + bw, by + bh],
          [mx, by + bh],
          [bx, by + bh],
          [bx, my],
        ];
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#1d4ed8';
        ctx.lineWidth = 1.5;
        for (const [hx, hy] of pts) {
          ctx.fillRect(hx - 4, hy - 4, 8, 8);
          ctx.strokeRect(hx - 4, hy - 4, 8, 8);
        }
      }
    }

    if (this.interaction.mode === 'draw') {
      const draft = clampNormalizedBox(
        this.interaction.drawStartNormX,
        this.interaction.drawStartNormY,
        this.interaction.drawCurNormX - this.interaction.drawStartNormX,
        this.interaction.drawCurNormY - this.interaction.drawStartNormY
      );
      ctx.save();
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(
        dr.x + draft.x * dr.w,
        dr.y + draft.y * dr.h,
        draft.width * dr.w,
        draft.height * dr.h
      );
      ctx.restore();
    }
  }

  private emptyInteraction(): InteractionState {
    return {
      mode: 'none',
      startClientX: 0,
      startClientY: 0,
      origPanX: 0,
      origPanY: 0,
      drawStartNormX: 0,
      drawStartNormY: 0,
      drawCurNormX: 0,
      drawCurNormY: 0,
      boxId: null,
      origBox: null,
      handle: null,
    };
  }
}
