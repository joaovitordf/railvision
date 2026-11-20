import { HttpErrorResponse } from '@angular/common/http';
import { ProblemDetail } from '../models/inspection.models';

export function extractApiErrorMessage(err: unknown, fallbackMessage: string): string {
  if (err instanceof HttpErrorResponse) {
    const body = err.error;
    if (body && typeof body === 'object') {
      const problem = body as ProblemDetail;
      if (problem.detail && problem.detail.trim().length > 0) {
        return problem.detail;
      }
      if (problem.title && problem.title.trim().length > 0) {
        return problem.title;
      }
    }
    if (typeof body === 'string' && body.trim().length > 0) {
      return body;
    }
    if (err.status === 0) {
      return 'Não foi possível conectar ao servidor backend local.';
    }
  }
  return fallbackMessage;
}
