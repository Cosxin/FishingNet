using log4net;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Buffers;

#nullable enable
namespace FishingFun
{
    public class ModelBobberFinderBiteWatcher : IBobberFinder, IImageProvider, IBiteWatcher
    {
        private static readonly ILog logger = LogManager.GetLogger("Fishbot");

        private Point previousLocation;
        private Point initBobberLocation;
        private bool isBite;
        private Bitmap bitmap = new Bitmap(1, 1);
        private static readonly float isBiteThreshold = 0.85f;
        private DateTime lastQueryTime = DateTime.MinValue;

        public event EventHandler<BobberBitmapEvent> BitmapEvent;

        public ModelBobberFinderBiteWatcher()
        {
            BitmapEvent += (s, e) => { };
        }

        public void Reset()
        {
            isBite = false;
            previousLocation = Point.Empty;
        }

        public void Reset(Point initialBobberPosition)
        {
            isBite = false;
            initBobberLocation = initialBobberPosition;
            RaiseEvent(new FishingEvent { Action = FishingAction.Reset });
        }

        public void RaiseEvent(FishingEvent ev)
        {
            FishingEventHandler?.Invoke(ev);
        }

        public bool IsBite(Point currentBobberPosition)
        {
            return isBite; // Additional bite detection logic can be implemented here
        }

        public Action<FishingEvent> FishingEventHandler { get; set; } = (e) => { };

        public Point Find()
        {
            if (IsQueryTooSoon())
            {
                return previousLocation;
            }

            lastQueryTime = DateTime.Now;
            Stopwatch watch = Stopwatch.StartNew();

            (Bitmap, long)[] sequence = new (Bitmap, long)[Model.SEQUENCE_LENGTH];
            int length = CaptureFrames(sequence);

            if (length != Model.SEQUENCE_LENGTH)
            {
                logger.Info("Warm start");
                return Point.Empty;
            }

            byte[] bmpData = ExtractBitmapData(sequence, length);
            float[] tensorData = NormalizeBitmapData(bmpData);

            PerformInference(tensorData, out (float, float, float) result);
            Point screenPosition = GetScreenPosition(result);
            
            UpdateState(result, screenPosition);

            watch.Stop();
            logger.Info("Total Inference Time: " + watch.ElapsedMilliseconds + "ms");

            return screenPosition;
        }

        private bool IsQueryTooSoon()
        {
            return DateTime.Now - lastQueryTime < TimeSpan.FromMilliseconds(200);
        }

        private int CaptureFrames((Bitmap, long)[] sequence)
        {
            ScreenRecorder.GetFrames(in sequence, out int length);
            return length;
        }

        private byte[] ExtractBitmapData((Bitmap, long)[] sequence, int length)
        {
            byte[] bmpData = new byte[length * 3 * 320 * 320];

            for (int frameIndex = 0; frameIndex < length; frameIndex++)
            {
                Bitmap frame = sequence[frameIndex].Item1;
                BitmapData data = frame.LockBits(new Rectangle(0, 0, 320, 320), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                int dataLength = data.Height * data.Stride;
                Marshal.Copy(data.Scan0, bmpData, dataLength * frameIndex, dataLength);
                frame.UnlockBits(data);
            }

            return bmpData;
        }

        private float[] NormalizeBitmapData(byte[] bmpData)
        {
            float[] tensorData = new float[bmpData.Length];

            for (int i = 0; i < bmpData.Length; i++)
            {
                tensorData[i] = bmpData[i] / 255.0f;
            }

            return tensorData;
        }

        private void PerformInference(float[] tensorData, out (float, float, float) result)
        {
            Model.Inference(tensorData, out result);
        }

        private Point GetScreenPosition((float, float, float) result)
        {
            Point bitmapPosition = new Point((int)(result.Item1 * 640), (int)(result.Item2 * 640));
            return WowScreen.GetScreenPositionFromBitmapPostion(bitmapPosition);
        }

        private void UpdateState((float, float, float) result, Point screenPosition)
        {
            isBite = (result.Item3 > isBiteThreshold);
            bitmap = WowScreen.GetBitmap();
            BitmapEvent?.Invoke(this, new BobberBitmapEvent { Point = new Point(screenPosition.X, screenPosition.Y), Bitmap = bitmap });
            previousLocation = screenPosition;
            bitmap.Dispose();
        }
    }
}