import { useRef, useState } from 'react'
import './App.css'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '')

function App() {
  const [file, setFile] = useState(null)
  const [status, setStatus] = useState('idle')
  const [uploadProgress, setUploadProgress] = useState(0)
  const [videoUrl, setVideoUrl] = useState('')
  const [error, setError] = useState('')
  const [logs, setLogs] = useState([])
  const fileInputRef = useRef(null)
  const isBusy = status === 'uploading' || status === 'processing'

  function addLog(message, type = 'info') {
    setLogs((current) => [
      ...current,
      {
        id: `${Date.now()}-${current.length}`,
        time: new Date().toLocaleTimeString(),
        message,
        type,
      },
    ])
  }

  function chooseFile(nextFile) {
    if (!nextFile) return

    setFile(nextFile)
    setVideoUrl('')
    setError('')
    setUploadProgress(0)
    setStatus('idle')
    setLogs([])
    addLog(`Selected ${nextFile.name}`)
  }

  function handleFileChange(event) {
    chooseFile(event.target.files?.[0])
  }

  function handleSubmit(event) {
    event.preventDefault()
    if (!file || isBusy) return

    setStatus('uploading')
    setVideoUrl('')
    setError('')
    setUploadProgress(0)
    addLog('Starting video upload')

    const formData = new FormData()
    formData.append('file', file)

    const request = new XMLHttpRequest()
    request.open('POST', `${API_URL}/api/upload`)
    request.timeout = 20 * 60 * 1000
    request.upload.addEventListener('progress', (progressEvent) => {
      if (!progressEvent.lengthComputable) return
      const percentage = Math.round(
        (progressEvent.loaded / progressEvent.total) * 100,
      )
      setUploadProgress(percentage)
    })
    request.upload.addEventListener('load', () => {
      setUploadProgress(100)
      setStatus('processing')
      addLog('Upload complete. Server is analyzing the video.')
    })
    request.addEventListener('load', () => {
      let response
      try {
        response = JSON.parse(request.responseText)
      } catch {
        setStatus('error')
        setError('The server returned an unreadable response.')
        addLog('Server returned an unreadable response', 'error')
        return
      }

      if (request.status < 200 || request.status >= 300) {
        const message =
          typeof response.detail === 'string'
            ? response.detail
            : `The server returned status ${request.status}.`
        setStatus('error')
        setError(message)
        addLog(`Analysis failed: ${message}`, 'error')
        return
      }

      if (!response.ai_video_url) {
        setStatus('error')
        setError('The analysis finished, but no video link was returned.')
        addLog('No result link was returned by the server', 'error')
        return
      }

      setVideoUrl(response.ai_video_url)
      setStatus('complete')
      addLog('Analysis complete. Result link is ready.', 'success')
    })
    request.addEventListener('error', () => {
      setStatus('error')
      const destination = API_URL || 'the current frontend origin'
      const message = `The browser could not complete the request to ${destination}. Check the backend address, CORS settings, and whether the frontend is being served over HTTPS while the backend uses HTTP.`
      setError(message)
      addLog('Network request failed before a server response was received', 'error')
    })
    request.addEventListener('timeout', () => {
      setStatus('error')
      const message =
        'The request timed out while the server was processing the video. Check the backend terminal and try again.'
      setError(message)
      addLog('The server did not finish before the 20-minute request limit', 'error')
    })
    request.addEventListener('abort', () => {
      setStatus('error')
      setError('The upload was interrupted.')
      addLog('Upload was interrupted', 'error')
    })
    request.send(formData)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>VAR Offside Detector</h1>
      </header>

      <section className="workspace" aria-label="Video analysis">
        <div className="primary-column">
          <section className="panel upload-panel" aria-labelledby="upload-title">
            <div className="panel-heading">
              <div>
                <h2 id="upload-title">Upload video</h2>
                <p>Choose a match clip to analyze.</p>
              </div>
            </div>

            <form onSubmit={handleSubmit}>
              <input
                ref={fileInputRef}
                className="file-input"
                type="file"
                accept="video/*,.mp4,.mov,.avi,.mkv"
                onChange={handleFileChange}
                disabled={isBusy}
                aria-label="Choose a video file"
              />
              <button
                className="file-picker"
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isBusy}
              >
                <span className="upload-icon" aria-hidden="true">
                  ↑
                </span>
                <span className="file-picker-copy">
                  <strong>{file ? 'Change video' : 'Choose a video'}</strong>
                  <span>MP4, MOV, AVI or another video format</span>
                </span>
              </button>

              {file && (
                <div className="selected-file" title={file.name}>
                  <span className="file-indicator" aria-hidden="true" />
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </span>
                </div>
              )}

              {status === 'uploading' && (
                <div className="upload-progress">
                  <div className="progress-label">
                    <span>Uploading</span>
                    <span>{uploadProgress}%</span>
                  </div>
                  <progress value={uploadProgress} max="100">
                    {uploadProgress}%
                  </progress>
                </div>
              )}

              {status === 'processing' && (
                <div className="processing-note" role="status">
                  <span className="spinner" aria-hidden="true" />
                  <span>Video uploaded. Analysis is running on the server…</span>
                </div>
              )}

              {error && (
                <p className="error-message" role="alert">
                  {error}
                </p>
              )}

              <button
                className="analyze-button"
                type="submit"
                disabled={!file || isBusy}
              >
                {status === 'uploading'
                  ? 'Uploading…'
                  : status === 'processing'
                    ? 'Analyzing…'
                    : status === 'complete'
                      ? 'Analyze another video'
                      : 'Upload and analyze'}
              </button>
            </form>
          </section>

          <section className="panel result-panel" aria-labelledby="result-title">
            <div className="panel-heading">
              <div>
                <h2 id="result-title">Result</h2>
                <p>Your processed video link will appear here.</p>
              </div>
            </div>

            {videoUrl ? (
              <div className="result-ready">
                <div className="result-status">
                  <span className="status-dot" aria-hidden="true" />
                  Analysis complete
                </div>
                <a
                  className="result-link"
                  href={videoUrl}
                  target="_blank"
                  rel="noreferrer"
                >
                  <span className="play-icon" aria-hidden="true">
                    ▶
                  </span>
                  Open analyzed video
                  <span className="external-icon" aria-hidden="true">
                    ↗
                  </span>
                </a>
                <video
                  className="result-video"
                  src={videoUrl}
                  controls
                  playsInline
                  preload="metadata"
                >
                  Your browser does not support video playback.
                </video>
              </div>
            ) : (
              <div className="result-empty">
                <span className="result-empty-icon" aria-hidden="true">
                  ▷
                </span>
                <span>
                  {status === 'processing'
                    ? 'The result will show here when analysis is finished.'
                    : 'No result yet'}
                </span>
              </div>
            )}
          </section>
        </div>

        <aside className="panel log-panel" aria-labelledby="log-title">
          <div className="panel-heading">
            <div>
              <h2 id="log-title">Progress</h2>
              <p>Upload and analysis status.</p>
            </div>
          </div>

          <div className="log-list" aria-live="polite" aria-relevant="additions">
            {logs.length ? (
              logs.map((log) => (
                <div className={`log-entry log-${log.type}`} key={log.id}>
                  <span className="log-marker" aria-hidden="true" />
                  <time>{log.time}</time>
                  <p>{log.message}</p>
                </div>
              ))
            ) : (
              <div className="log-empty">
                <span className="log-pulse" aria-hidden="true" />
                <p>Choose a video to start.</p>
              </div>
            )}
            {status === 'processing' && (
              <div className="log-entry log-active">
                <span className="log-marker" aria-hidden="true" />
                <time>Now</time>
                <p>Waiting for the server to finish analysis…</p>
              </div>
            )}
          </div>
          <p className="log-footnote">
            Frame-by-frame processing details are shown in the backend terminal.
          </p>
        </aside>
      </section>
    </main>
  )
}

export default App
