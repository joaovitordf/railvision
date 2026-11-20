import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { extractApiErrorMessage } from './api-error.util';
import { RailvisionApiService } from './railvision-api.service';

describe('RailvisionApiService', () => {
  let service: RailvisionApiService;
  let httpMock: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(RailvisionApiService);
    httpMock = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    httpMock.verify();
  });

  it('uploads video via FormData with title and notes', () => {
    const file = new File(['mock-video'], 'yard.mp4', { type: 'video/mp4' });
    let capturedId = '';

    service.createInspection(file, 'Patio Sul', 'Camera 02').subscribe((res) => {
      capturedId = res.id;
    });

    const req = httpMock.expectOne('/api/v1/inspections');
    expect(req.request.method).toBe('POST');
    expect(req.request.body instanceof FormData).toBe(true);
    const formData = req.request.body as FormData;
    expect(formData.get('title')).toBe('Patio Sul');
    expect(formData.get('notes')).toBe('Camera 02');
    expect(formData.get('file')).toBeTruthy();

    req.flush({
      id: 'insp-1',
      title: 'Patio Sul',
      notes: 'Camera 02',
      status: 'UPLOADED',
      originalFilename: 'yard.mp4',
      contentType: 'video/mp4',
      fileSizeBytes: 10,
      sha256Checksum: 'abc',
      latestJobId: null,
      createdAt: '2026-11-19T00:00:00Z',
      updatedAt: '2026-11-19T00:00:00Z',
    });

    expect(capturedId).toBe('insp-1');
  });

  it('triggers processing and polls job status', () => {
    let status = '';
    service.startProcessing('insp-1').subscribe((job) => {
      status = job.status;
    });

    const processReq = httpMock.expectOne('/api/v1/inspections/insp-1/process');
    expect(processReq.request.method).toBe('POST');
    processReq.flush({
      id: 'job-1',
      inspectionId: 'insp-1',
      status: 'QUEUED',
      progressPercent: 0,
      engineType: 'DEMO_FIXTURE_ENGINE',
      isDemoSimulation: true,
      errorCode: null,
      errorMessage: null,
      startedAt: null,
      finishedAt: null,
      createdAt: '2026-11-19T00:00:00Z',
    });
    expect(status).toBe('QUEUED');

    service.getJob('job-1').subscribe((job) => {
      status = job.status;
    });
    const jobReq = httpMock.expectOne('/api/v1/jobs/job-1');
    expect(jobReq.request.method).toBe('GET');
    jobReq.flush({
      id: 'job-1',
      inspectionId: 'insp-1',
      status: 'COMPLETED',
      progressPercent: 100,
      engineType: 'DEMO_FIXTURE_ENGINE',
      isDemoSimulation: true,
      errorCode: null,
      errorMessage: null,
      startedAt: '2026-11-19T00:00:01Z',
      finishedAt: '2026-11-19T00:00:02Z',
      createdAt: '2026-11-19T00:00:00Z',
    });
    expect(status).toBe('COMPLETED');
  });

  it('extracts RFC 9457 ProblemDetail detail message', () => {
    const httpError = new HttpErrorResponse({
      status: 400,
      error: {
        title: 'Mídia inválida',
        status: 400,
        detail: 'O conteúdo do arquivo não corresponde a um container de vídeo válido.',
      },
    });

    expect(extractApiErrorMessage(httpError, 'Erro genérico')).toBe(
      'O conteúdo do arquivo não corresponde a um container de vídeo válido.'
    );
  });
});
