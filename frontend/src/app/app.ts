import { Component, OnDestroy } from '@angular/core';
import { finalize } from 'rxjs';

import {
  CatTranslatorService,
  CatTranslationResponse,
} from './cat-translator.service';

type TranslatorStatus = 'Ready' | 'Recording' | 'Translating';
type ResultState = 'confident' | 'uncertain' | 'unknown';

@Component({
  selector: 'app-root',
  imports: [],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnDestroy {
  status: TranslatorStatus = 'Ready';
  isRecording = false;
  isLoading = false;
  errorMessage = '';
  result: CatTranslationResponse | null = null;

  private mediaRecorder: MediaRecorder | null = null;
  private mediaStream: MediaStream | null = null;
  private audioChunks: Blob[] = [];
  private recordingTimerId: number | null = null;

  constructor(private readonly catTranslatorService: CatTranslatorService) {}

  get canRecord(): boolean {
    return !this.isRecording && !this.isLoading;
  }

  get confidencePercent(): number {
    return this.result ? Math.round(this.result.confidence * 100) : 0;
  }

  get resultState(): ResultState {
    if (this.result?.intent === 'Uncertain') {
      return 'uncertain';
    }

    if (this.result?.intent === 'Unknown') {
      return 'unknown';
    }

    return 'confident';
  }

  get resultKicker(): string {
    if (this.resultState === 'uncertain') {
      return 'Uncertain result';
    }

    if (this.resultState === 'unknown') {
      return 'No clear intent found';
    }

    return 'Predicted intent';
  }

  get displayedIntent(): string {
    if (!this.result) {
      return '';
    }

    if (this.result.intent === 'Uncertain') {
      return `Maybe ${this.result.top_guess}`;
    }

    return this.result.intent;
  }

  predictionPercent(confidence: number): number {
    return Math.round(confidence * 100);
  }

  async startRecording(): Promise<void> {
    if (!this.canRecord) {
      return;
    }

    this.errorMessage = '';
    this.result = null;
    this.audioChunks = [];

    if (!navigator.mediaDevices?.getUserMedia) {
      this.errorMessage = 'This browser does not support microphone recording.';
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      this.errorMessage = 'This browser does not support the MediaRecorder API.';
      return;
    }

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = this.getSupportedMimeType();
      const recorderOptions = mimeType ? { mimeType } : undefined;

      this.mediaRecorder = new MediaRecorder(this.mediaStream, recorderOptions);

      this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onerror = () => {
        this.errorMessage = 'Recording failed. Please try again.';
        this.resetRecorder();
        this.status = 'Ready';
        this.isRecording = false;
      };

      this.mediaRecorder.onstop = () => {
        const recording = new Blob(this.audioChunks, {
          type: this.mediaRecorder?.mimeType || 'audio/webm',
        });

        this.resetRecorder();
        this.uploadRecording(recording);
      };

      this.status = 'Recording';
      this.isRecording = true;
      this.mediaRecorder.start();

      this.recordingTimerId = window.setTimeout(() => {
        this.stopRecording();
      }, 3000);
    } catch (error) {
      this.resetRecorder();
      this.status = 'Ready';
      this.isRecording = false;
      this.errorMessage = this.microphoneErrorMessage(error);
    }
  }

  ngOnDestroy(): void {
    this.resetRecorder();
  }

  private stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
  }

  private uploadRecording(recording: Blob): void {
    this.isRecording = false;

    if (recording.size === 0) {
      this.status = 'Ready';
      this.errorMessage = 'No audio was captured. Please try recording again.';
      return;
    }

    this.status = 'Translating';
    this.isLoading = true;

    this.catTranslatorService
      .translateAudio(recording)
      .pipe(
        finalize(() => {
          this.isLoading = false;
          this.status = 'Ready';
        }),
      )
      .subscribe({
        next: (response) => {
          this.result = response;
        },
        error: (error) => {
          console.error('Cat translator backend error:', error);
          this.errorMessage =
            error?.error?.detail ||
            error?.message ||
            'Could not predict this recording. Try again with a clear cat sound.';
        },
      });
  }

  private getSupportedMimeType(): string {
    const preferredTypes = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/ogg',
    ];

    return preferredTypes.find((type) => MediaRecorder.isTypeSupported(type)) || '';
  }

  private resetRecorder(): void {
    if (this.recordingTimerId !== null) {
      window.clearTimeout(this.recordingTimerId);
      this.recordingTimerId = null;
    }

    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.mediaRecorder = null;
  }

  private microphoneErrorMessage(error: unknown): string {
    if (error instanceof DOMException && error.name === 'NotAllowedError') {
      return 'Microphone permission was blocked. Please allow microphone access and try again.';
    }

    if (error instanceof DOMException && error.name === 'NotFoundError') {
      return 'No microphone was found. Please connect one and try again.';
    }

    return 'Could not start recording. Please check your microphone and browser permissions.';
  }
}
