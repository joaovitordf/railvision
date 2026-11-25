import { TestBed } from '@angular/core/testing';
import { TrainingEpochPoint, TrainingLineagePoint } from '../models/training.models';
import { TrainingChartComponent } from './training-chart.component';

describe('TrainingChartComponent', () => {
  it('renders sequential lineage polylines and validation-change marker', () => {
    const lineage: TrainingLineagePoint[] = [
      {
        versionLabel: 'v0',
        runId: null,
        experimentName: 'exp04c',
        status: 'BASELINE',
        datasetId: 'open-top-yolo-v1',
        valFingerprint: 'fp1',
        valDatasetChanged: false,
        baseLabel: 'yolo26n.pt',
        epochsCompleted: 60,
        map50: 0.9,
        map5095: 0.63,
        precision: 0.84,
        recall: 0.91,
        bestF1Conf: null,
        registeredModelId: 'exp04c_yolo26n_opentop_640',
        notes: null,
        createdAt: null,
      },
      {
        versionLabel: 'v1',
        runId: 'run-1',
        experimentName: 'train_v1',
        status: 'COMPLETED',
        datasetId: 'open-top-yolo-v2',
        valFingerprint: 'fp2',
        valDatasetChanged: true,
        baseLabel: 'v0',
        epochsCompleted: 30,
        map50: 0.78,
        map5095: 0.54,
        precision: 0.81,
        recall: 0.76,
        bestF1Conf: 0.35,
        registeredModelId: null,
        notes: 'Multicampanha',
        createdAt: '2026-11-24T19:00:00Z',
      },
    ];

    const fixture = TestBed.createComponent(TrainingChartComponent);
    fixture.componentRef.setInput('mode', 'lineage');
    fixture.componentRef.setInput('lineagePoints', lineage);
    fixture.componentRef.setInput('baselineMap50', 0.9);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('svg polyline').length).toBe(3);
    expect(el.textContent).toContain('Novo val');
    expect(el.textContent).toContain('Ref v0 (90.0%)');
    expect(el.textContent).toContain('v1');
  });

  it('switches between epoch metrics and losses views', () => {
    const epochs: TrainingEpochPoint[] = [
      {
        epoch: 1,
        timeSec: 2.1,
        trainBoxLoss: 1.8,
        trainClsLoss: 2.5,
        trainL1Loss: 0.02,
        valBoxLoss: 1.7,
        valClsLoss: 2.3,
        valL1Loss: 0.02,
        precision: 0.65,
        recall: 0.61,
        map50: 0.68,
        map5095: 0.42,
        lr: 0.0005,
      },
      {
        epoch: 2,
        timeSec: 4.2,
        trainBoxLoss: 1.4,
        trainClsLoss: 1.9,
        trainL1Loss: 0.01,
        valBoxLoss: 1.3,
        valClsLoss: 1.8,
        valL1Loss: 0.015,
        precision: 0.78,
        recall: 0.74,
        map50: 0.81,
        map5095: 0.53,
        lr: 0.0004,
      },
    ];

    const fixture = TestBed.createComponent(TrainingChartComponent);
    fixture.componentRef.setInput('mode', 'epochs');
    fixture.componentRef.setInput('epochMetricMode', 'metrics');
    fixture.componentRef.setInput('epochHistory', epochs);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelectorAll('svg polyline').length).toBe(4);
    expect(el.textContent).toContain('mAP50 81.0%');

    fixture.componentRef.setInput('epochMetricMode', 'losses');
    fixture.detectChanges();
    expect(el.textContent).toContain('Train Box');
    expect(el.textContent).toContain('val_box 1.300');
  });
});
