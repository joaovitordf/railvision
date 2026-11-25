import { DecimalPipe } from '@angular/common';
import { Component, computed, input, output, signal } from '@angular/core';
import { TrainingEpochPoint, TrainingLineagePoint } from '../models/training.models';

export type TrainingChartMode = 'lineage' | 'epochs';
export type EpochMetricMode = 'metrics' | 'losses';

interface ChartSeriesLine {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
  pointsAttr: string;
  coords: { x: number; y: number; value: number; rawLabel: string }[];
}

const PAD_L = 48;
const PAD_R = 20;
const PAD_T = 18;
const PAD_B = 32;
const VIEW_W = 640;
const VIEW_H = 252;
const PLOT_W = VIEW_W - PAD_L - PAD_R;
const PLOT_H = VIEW_H - PAD_T - PAD_B;

@Component({
  selector: 'rv-training-chart',
  standalone: true,
  imports: [DecimalPipe],
  template: `
    <div class="rv-chart-wrap">
      @if (!hasData()) {
        <div class="rv-chart-empty">
          Aguardando dados suficientes para desenhar o gráfico.
        </div>
      } @else {
        <svg
          class="rv-chart-svg"
          [attr.viewBox]="'0 0 ' + viewW + ' ' + viewH"
          role="img"
          [attr.aria-label]="ariaLabel()"
        >
          <!-- Horizontal grid & Y axis labels -->
          @for (tick of yTicks(); track tick.y) {
            <line
              [attr.x1]="padL"
              [attr.x2]="viewW - padR"
              [attr.y1]="tick.y"
              [attr.y2]="tick.y"
              stroke="var(--rv-border)"
              stroke-width="1"
              stroke-dasharray="3 3"
            />
            <text
              [attr.x]="padL - 8"
              [attr.y]="tick.y + 4"
              text-anchor="end"
              fill="var(--rv-text-muted)"
              font-size="11"
              font-family="JetBrains Mono, monospace"
            >
              {{ tick.label }}
            </text>
          }

          <!-- Baseline reference line -->
          @if (baselineLineY(); as baseY) {
            <line
              [attr.x1]="padL"
              [attr.x2]="viewW - padR"
              [attr.y1]="baseY"
              [attr.y2]="baseY"
              stroke="#64748b"
              stroke-width="1.4"
              stroke-dasharray="6 4"
            />
            <text
              [attr.x]="viewW - padR - 4"
              [attr.y]="baseY - 5"
              text-anchor="end"
              fill="var(--rv-text-secondary)"
              font-size="10.5"
              font-weight="600"
            >
              Ref v0 ({{ baselineMap50()! * 100 | number: '1.1-1' }}%)
            </text>
          }

          <!-- Validation change vertical markers in lineage mode -->
          @for (marker of valChangeMarkers(); track marker.x) {
            <line
              [attr.x1]="marker.x"
              [attr.x2]="marker.x"
              [attr.y1]="padT"
              [attr.y2]="viewH - padB"
              stroke="#d97706"
              stroke-width="1.5"
              stroke-dasharray="4 3"
            />
            <rect
              [attr.x]="marker.x - 28"
              [attr.y]="padT"
              width="56"
              height="16"
              rx="4"
              fill="var(--rv-warn-bg)"
            />
            <text
              [attr.x]="marker.x"
              [attr.y]="padT + 11.5"
              text-anchor="middle"
              fill="var(--rv-warn-text)"
              font-size="9.5"
              font-weight="700"
            >
              Novo val
            </text>
          }

          <!-- X axis labels -->
          @for (xt of xTicks(); track xt.index) {
            <text
              [attr.x]="xt.x"
              [attr.y]="viewH - 10"
              text-anchor="middle"
              fill="var(--rv-text-muted)"
              font-size="11"
              font-family="JetBrains Mono, monospace"
            >
              {{ xt.label }}
            </text>
          }

          <!-- Plotted lines -->
          @for (line of seriesLines(); track line.key) {
            <polyline
              fill="none"
              [attr.stroke]="line.color"
              [attr.stroke-width]="line.dashed ? 1.8 : 2.4"
              [attr.stroke-dasharray]="line.dashed ? '5 4' : null"
              stroke-linecap="round"
              stroke-linejoin="round"
              [attr.points]="line.pointsAttr"
            />
            @for (pt of line.coords; track $index) {
              <circle
                [attr.cx]="pt.x"
                [attr.cy]="pt.y"
                [attr.r]="hoverIndex() === $index ? 5.2 : (mode() === 'lineage' ? 4.2 : 2.6)"
                [attr.fill]="line.color"
                stroke="var(--rv-bg-surface)"
                stroke-width="1.5"
                style="cursor: pointer;"
                (mouseenter)="hoverIndex.set($index)"
                (click)="onPointClick($index)"
              />
            }
          }
        </svg>

        <!-- Interactive Legend & Readout Bar -->
        <div class="rv-chart-footer">
          <div class="rv-chart-legend">
            @for (line of seriesLines(); track line.key) {
              <span class="rv-chart-legend__item">
                <span
                  class="rv-chart-swatch"
                  [style.background]="line.dashed ? 'transparent' : line.color"
                  [style.border]="line.dashed ? '2px dashed ' + line.color : 'none'"
                ></span>
                <span>{{ line.label }}</span>
              </span>
            }
          </div>

          @if (activeReadout(); as info) {
            <div class="rv-chart-readout rv-mono">{{ info }}</div>
          }
        </div>
      }
    </div>
  `,
  styles: [
    `
      .rv-chart-wrap {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }

      .rv-chart-svg {
        width: 100%;
        height: auto;
        display: block;
        overflow: visible;
      }

      .rv-chart-empty {
        height: 210px;
        display: grid;
        place-items: center;
        border: 1px dashed var(--rv-border);
        border-radius: var(--rv-radius-sm);
        color: var(--rv-text-muted);
        font-size: 0.85rem;
      }

      .rv-chart-footer {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px 14px;
        padding-top: 4px;
        border-top: 1px solid var(--rv-border);
      }

      .rv-chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        font-size: 0.78rem;
        color: var(--rv-text-secondary);
      }

      .rv-chart-legend__item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .rv-chart-swatch {
        width: 12px;
        height: 3px;
        border-radius: 2px;
        display: inline-block;
      }

      .rv-chart-readout {
        font-size: 0.76rem;
        color: var(--rv-text-primary);
      }
    `,
  ],
})
export class TrainingChartComponent {
  readonly mode = input.required<TrainingChartMode>();
  readonly epochMetricMode = input<EpochMetricMode>('metrics');
  readonly lineagePoints = input<TrainingLineagePoint[]>([]);
  readonly epochHistory = input<TrainingEpochPoint[]>([]);
  readonly compareEpochHistory = input<TrainingEpochPoint[]>([]);
  readonly compareLabel = input<string | null>(null);
  readonly baselineMap50 = input<number | null>(null);
  readonly ariaLabel = input<string>('Gráfico de treinamento');

  readonly lineageSelected = output<TrainingLineagePoint>();

  protected readonly viewW = VIEW_W;
  protected readonly viewH = VIEW_H;
  protected readonly padL = PAD_L;
  protected readonly padR = PAD_R;
  protected readonly padT = PAD_T;
  protected readonly padB = PAD_B;

  protected readonly hoverIndex = signal<number | null>(null);

  protected readonly hasData = computed(() =>
    this.mode() === 'lineage' ? this.lineagePoints().length > 0 : this.epochHistory().length > 0
  );

  private readonly yDomain = computed<{ min: number; max: number; isPercent: boolean }>(() => {
    if (this.mode() === 'lineage' || this.epochMetricMode() === 'metrics') {
      return { min: 0, max: 1, isPercent: true };
    }
    const vals: number[] = [];
    for (const p of this.epochHistory()) {
      vals.push(p.trainBoxLoss, p.valBoxLoss, p.trainClsLoss, p.valClsLoss);
    }
    const maxVal = Math.max(1.0, ...vals);
    return { min: 0, max: Number((maxVal * 1.1).toFixed(2)), isPercent: false };
  });

  protected readonly yTicks = computed(() => {
    const dom = this.yDomain();
    const steps = 4;
    const ticks: { y: number; label: string }[] = [];
    for (let i = 0; i <= steps; i++) {
      const frac = i / steps;
      const val = dom.min + (dom.max - dom.min) * frac;
      const y = PAD_T + PLOT_H - frac * PLOT_H;
      ticks.push({
        y,
        label: dom.isPercent ? `${Math.round(val * 100)}%` : val.toFixed(1),
      });
    }
    return ticks;
  });

  protected readonly baselineLineY = computed<number | null>(() => {
    const ref = this.baselineMap50();
    const dom = this.yDomain();
    if (ref == null || !dom.isPercent) {
      return null;
    }
    return PAD_T + PLOT_H - Math.max(0, Math.min(1, ref)) * PLOT_H;
  });

  private xPos(index: number, count: number): number {
    if (count <= 1) {
      return PAD_L + PLOT_W / 2;
    }
    return PAD_L + (index / (count - 1)) * PLOT_W;
  }

  private yPos(val: number): number {
    const dom = this.yDomain();
    const span = Math.max(1e-6, dom.max - dom.min);
    const norm = Math.max(0, Math.min(1, (val - dom.min) / span));
    return PAD_T + PLOT_H - norm * PLOT_H;
  }

  protected readonly xTicks = computed(() => {
    if (this.mode() === 'lineage') {
      const pts = this.lineagePoints();
      return pts.map((p, idx) => ({
        index: idx,
        x: this.xPos(idx, pts.length),
        label: p.versionLabel,
      }));
    }
    const pts = this.epochHistory();
    const n = pts.length;
    if (n === 0) return [];
    const stride = n <= 8 ? 1 : Math.ceil(n / 7);
    const out: { index: number; x: number; label: string }[] = [];
    for (let i = 0; i < n; i++) {
      if (i === 0 || i === n - 1 || i % stride === 0) {
        out.push({ index: i, x: this.xPos(i, n), label: `E${pts[i].epoch}` });
      }
    }
    return out;
  });

  protected readonly valChangeMarkers = computed(() => {
    if (this.mode() !== 'lineage') return [];
    const pts = this.lineagePoints();
    const markers: { x: number; label: string }[] = [];
    pts.forEach((p, idx) => {
      if (p.valDatasetChanged) {
        markers.push({ x: this.xPos(idx, pts.length), label: p.datasetId });
      }
    });
    return markers;
  });

  protected readonly seriesLines = computed<ChartSeriesLine[]>(() => {
    if (this.mode() === 'lineage') {
      const pts = this.lineagePoints();
      const specs: { key: string; label: string; color: string; pick: (p: TrainingLineagePoint) => number | null }[] = [
        { key: 'map50', label: 'mAP50', color: '#2563eb', pick: (p) => p.map50 },
        { key: 'map5095', label: 'mAP50-95', color: '#16a34a', pick: (p) => p.map5095 },
        { key: 'recall', label: 'Recall', color: '#d97706', pick: (p) => p.recall },
      ];
      return specs
        .map((sp) => {
          const coords: { x: number; y: number; value: number; rawLabel: string }[] = [];
          pts.forEach((p, idx) => {
            const v = sp.pick(p);
            if (v != null) {
              coords.push({
                x: this.xPos(idx, pts.length),
                y: this.yPos(v),
                value: v,
                rawLabel: p.versionLabel,
              });
            }
          });
          return {
            key: sp.key,
            label: sp.label,
            color: sp.color,
            pointsAttr: coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' '),
            coords,
          };
        })
        .filter((s) => s.coords.length > 0);
    }

    const eps = this.epochHistory();
    const n = eps.length;
    if (this.epochMetricMode() === 'losses') {
      const specs: { key: string; label: string; color: string; dashed?: boolean; pick: (e: TrainingEpochPoint) => number }[] = [
        { key: 'tbox', label: 'Train Box', color: '#2563eb', pick: (e) => e.trainBoxLoss },
        { key: 'vbox', label: 'Val Box', color: '#f97316', dashed: true, pick: (e) => e.valBoxLoss },
        { key: 'tcls', label: 'Train Cls', color: '#7c3aed', pick: (e) => e.trainClsLoss },
        { key: 'vcls', label: 'Val Cls', color: '#ef4444', dashed: true, pick: (e) => e.valClsLoss },
      ];
      return specs.map((sp) => {
        const coords = eps.map((e, idx) => {
          const v = sp.pick(e);
          return { x: this.xPos(idx, n), y: this.yPos(v), value: v, rawLabel: `Época ${e.epoch}` };
        });
        return {
          key: sp.key,
          label: sp.label,
          color: sp.color,
          dashed: sp.dashed,
          pointsAttr: coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' '),
          coords,
        };
      });
    }

    const metricSpecs: { key: string; label: string; color: string; pick: (e: TrainingEpochPoint) => number }[] = [
      { key: 'map50', label: 'mAP50', color: '#2563eb', pick: (e) => e.map50 },
      { key: 'map5095', label: 'mAP50-95', color: '#16a34a', pick: (e) => e.map5095 },
      { key: 'precision', label: 'Precisão', color: '#7c3aed', pick: (e) => e.precision },
      { key: 'recall', label: 'Recall', color: '#d97706', pick: (e) => e.recall },
    ];
    const lines: ChartSeriesLine[] = metricSpecs.map((sp) => {
      const coords = eps.map((e, idx) => {
        const v = sp.pick(e);
        return { x: this.xPos(idx, n), y: this.yPos(v), value: v, rawLabel: `Época ${e.epoch}` };
      });
      return {
        key: sp.key,
        label: sp.label,
        color: sp.color,
        pointsAttr: coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' '),
        coords,
      };
    });

    const cmp = this.compareEpochHistory();
    if (cmp.length > 0) {
      const cmpCoords = cmp.map((e, idx) => ({
        x: this.xPos(idx, Math.max(n, cmp.length)),
        y: this.yPos(e.map50),
        value: e.map50,
        rawLabel: `Época ${e.epoch}`,
      }));
      lines.push({
        key: 'cmp_map50',
        label: `${this.compareLabel() || 'Anterior'} (mAP50)`,
        color: '#06b6d4',
        dashed: true,
        pointsAttr: cmpCoords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(' '),
        coords: cmpCoords,
      });
    }
    return lines;
  });

  protected readonly activeReadout = computed<string | null>(() => {
    if (this.mode() === 'lineage') {
      const pts = this.lineagePoints();
      if (!pts.length) return null;
      const idx = this.hoverIndex() != null && this.hoverIndex()! < pts.length ? this.hoverIndex()! : pts.length - 1;
      const p = pts[idx];
      const m50 = p.map50 != null ? `${(p.map50 * 100).toFixed(1)}%` : '—';
      const m95 = p.map5095 != null ? `${(p.map5095 * 100).toFixed(1)}%` : '—';
      const rec = p.recall != null ? `${(p.recall * 100).toFixed(1)}%` : '—';
      return `${p.versionLabel} · mAP50 ${m50} · mAP50-95 ${m95} · Recall ${rec}`;
    }

    const eps = this.epochHistory();
    if (!eps.length) return null;
    const idx = this.hoverIndex() != null && this.hoverIndex()! < eps.length ? this.hoverIndex()! : eps.length - 1;
    const e = eps[idx];
    if (this.epochMetricMode() === 'losses') {
      return `Época ${e.epoch} · train_box ${e.trainBoxLoss.toFixed(3)} · val_box ${e.valBoxLoss.toFixed(3)}`;
    }
    return `Época ${e.epoch} · mAP50 ${(e.map50 * 100).toFixed(1)}% · mAP50-95 ${(e.map5095 * 100).toFixed(1)}% · P ${(e.precision * 100).toFixed(1)}% · R ${(e.recall * 100).toFixed(1)}%`;
  });

  protected onPointClick(index: number): void {
    this.hoverIndex.set(index);
    if (this.mode() === 'lineage') {
      const pt = this.lineagePoints()[index];
      if (pt) {
        this.lineageSelected.emit(pt);
      }
    }
  }
}
