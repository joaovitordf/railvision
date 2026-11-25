import { Component } from '@angular/core';

@Component({
  selector: 'rv-footer',
  standalone: true,
  template: `
    <footer class="rv-footer">
      <span><strong>RailVision AI</strong> · Inspeção e Visão Computacional Ferroviária</span>
      <span>Ambiente local isolado · PyTorch CUDA / PyAV / FastAPI</span>
    </footer>
  `,
})
export class FooterComponent {}
