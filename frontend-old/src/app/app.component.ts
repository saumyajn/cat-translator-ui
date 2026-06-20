import { Component, OnDestroy } from '@angular/core';
import { finalize } from 'rxjs';

import {
  CatTranslationResponse,
  CatTranslatorService,
} from './cat-translator.service';

type RecorderStatus = 'Ready' | 'Recording' | 'Translating';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
  title = 'Cat Voice Translator';
  status: RecorderStatus = 'Ready';
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

  predictionPercent(confidence: number): number {
    return Math.round(confidence * 100);
  }

  async recordCatSound(): Promise<void> {
    if (!this.canRecord) {
      return;
    }

    this.errorMessage = '';
    this.result = null;
    this.audioChunks = [];

    if (!navigator.mediaDevices?.getUserMedia) {
      this.errorMessage = 'Your browser does not support microphone recording.';
      return;
    }

    if (typeof MediaRecorder === 'undefined') {
      this.errorMessage = 'Your browser does not support the MediaRecorder API.';
      return;
    }

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = this.getSupportedMimeType();
      const options = mimeType ? { mimeType } : undefined;

      this.mediaRecorder = new MediaRecorder(this.mediaStream, options);

      this.mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onerror = () => {
        this.finishRecordingSession();
        this.status = 'Ready';
        this.isRecording = false;
        this.errorMessage = 'Recording failed. Please try again.';
      };

      this.mediaRecorder.onstop = () => {
        const recording = new Blob(this.audioChunks, {
          type: this.mediaRecorder?.mimeType || 'audio/webm',
        });

        this.finishRecordingSession();
        this.sendRecording(recording);
      };

      this.status = 'Recording';
      this.isRecording = true;
      this.mediaRecorder.start();

      this.recordingTimerId = window.setTimeout(() => {
        this.stopRecording();
      }, 3000);
    } catch (error) {
      this.finishRecordingSession();
      this.status = 'Ready';
      this.isRecording = false;
      this.errorMessage = this.getMicrophoneErrorMessage(error);
    }
  }

  ngOnDestroy(): void {
    this.finishRecordingSession();
  }

  private stopRecording(): void {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
    }
  }

  private sendRecording(recording: Blob): void {
    if (recording.size === 0) {
      this.status = 'Ready';
      this.isRecording = false;
      this.errorMessage = 'No audio was captured. Please try recording again.';
      return;
    }

    this.status = 'Translating';
    this.isRecording = false;
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
          this.errorMessage =
            error?.error?.detail ||
            'Could not predict your cat sound. Try a clearer 3-second recording.';
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

  private finishRecordingSession(): void {
    if (this.recordingTimerId !== null) {
      window.clearTimeout(this.recordingTimerId);
      this.recordingTimerId = null;
    }

    this.mediaStream?.getTracks().forEach((track) => track.stop());
    this.mediaStream = null;
    this.mediaRecorder = null;
  }

  private getMicrophoneErrorMessage(error: unknown): string {
    if (error instanceof DOMException && error.name === 'NotAllowedError') {
      return 'Microphone permission was blocked. Please allow microphone access and try again.';
    }

    if (error instanceof DOMException && error.name === 'NotFoundError') {
      return 'No microphone was found. Please connect a microphone and try again.';
    }

    return 'Could not start recording. Please check your microphone and browser permissions.';
  }
}
