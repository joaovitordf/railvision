import { Component } from '@angular/core';

@Component({
  selector: 'rv-footer',
  standalone: true,
  template: `
    <footer class="rv-footer">
      <span><strong>RailVision AI</strong> · Sistema local de monitoramento ferroviário</span>
      <span>Python + FastAPI · PyTorch CUDA / ONNX / PyAV · Inferência local real</span>
    </footer>
  `,
})
export class FooterComponent {}
