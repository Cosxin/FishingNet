using log4net;
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;

namespace FishingFun
{
#nullable enable
    public static class ScreenRecorder
    {
        private static ILog logger = LogManager.GetLogger("Fishbot");
        private static int _frameRate = 8;
        private static int _bufferSize = 1;
        private static ConcurrentQueue<(Bitmap, long)> _frameBuffer;
        private static Thread? _recordingThread;
        private static bool _isRecording;
        private static readonly Object __lock = new object();

        static ScreenRecorder()
        {
            _frameBuffer = new ConcurrentQueue<(Bitmap, long)>();
            _recordingThread = null;
        }

        public static void SetFrameRate(int frameRate) { _frameRate = frameRate; }
        public static void SetBufferSize(int bufferSize) { _bufferSize = bufferSize; }

        public static void StartLoop()
        {
            if (!_isRecording && _recordingThread == null)
            {
                _isRecording = true;
                _recordingThread = new Thread(RecordScreen);
                _recordingThread.Start();
            }
        }
        public static void StopLoop()
        {
            if (_isRecording && _recordingThread != null)
            {
                _isRecording = false;
                if (_recordingThread.IsAlive)
                    _recordingThread.Join();
                _recordingThread = null;
            }
        }

        public static void GetFrames(in (Bitmap, long)[] buffer, out int length)
        {
            lock(__lock)
            {
                _frameBuffer.CopyTo(buffer, 0);
                length = _frameBuffer.Count;
            }
        }

        private static void RecordScreen()
        {
            _isRecording = true;

            logger.Info("Screen Thread Started");

            while (_isRecording)
            {
                var stopwatch = Stopwatch.StartNew();

                var screenshot = WowScreen.GetBitmap();

                var resizedScreenshot = new Bitmap(screenshot, new Size(320, 320));

                screenshot.Dispose(); 

                long now = DateTimeOffset.Now.ToUnixTimeMilliseconds();

                lock(__lock)
                {
                    _frameBuffer.Enqueue((resizedScreenshot, now));
                    if (_frameBuffer.Count > _bufferSize)
                    {
                        _frameBuffer.TryDequeue(out var _);
                    }
                }
               
                // Sleep to maintain frame rate
                var elapsedTime = stopwatch.ElapsedMilliseconds;
                var delayMs = (1000 / _frameRate) - (int)elapsedTime;
                
                if (delayMs > 0)
                {
                    Thread.Sleep(delayMs);
                }
            }
        }

        public static void SaveRecords(Point clicked_coords, string savePath)
        {
            string folderPath = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
            string fullFolderPath = Path.Combine(savePath, folderPath);

            if (!Directory.Exists(fullFolderPath))
                Directory.CreateDirectory(fullFolderPath);

            if (_isRecording) return;

            if (clicked_coords.X > 0)
            {
                string filename = "label.txt";
                string filePath = Path.Combine(fullFolderPath, filename);
                using (StreamWriter writer = new StreamWriter(filePath))
                {
                    writer.Write($"{clicked_coords.X},{clicked_coords.Y}");
                }
            }

            lock (__lock)
            {
                while (_frameBuffer.TryDequeue(out var frame))
                {
                    var (screenshot, timestamp) = frame;
                    // Save the screenshot
                    string fileName = $"screenshot_{timestamp}.png";
                    string filePath = Path.Combine(fullFolderPath, fileName);
                    Console.WriteLine("Saving records to " + filePath);
                    screenshot.Save(filePath, ImageFormat.Png);
                }
            }
        }

        public static void EnsureEmpty()
        {
            while (_frameBuffer.TryDequeue(out var frame))
            {
                frame.Item1.Dispose();
            }
        }
    }
}