using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using log4net;

namespace FishingFun
{
    
    public static class Model
    {
        private static ILog logger = LogManager.GetLogger("Fishbot");

        public static readonly int SEQUENCE_LENGTH = 3;

        private static InferenceSession _session;

        private static RunOptions _runOptions;

        private static readonly long[] inputShape = {1, SEQUENCE_LENGTH, 320, 320, 3 };

        private static DateTime lastInferenceTime;

        private static (float, float, float) lastInferenceValue;

        static Model()
        {
            _session = new InferenceSession("models/cnnlstm_320x320_epoch15.onnx");
            _runOptions = new RunOptions();
            float[] inital_sequence = new float[SEQUENCE_LENGTH * 320 * 320 * 3];
            Inference(inital_sequence, out (float, float, float) _);
        }

        public static void Inference(float[] inputData, out (float x, float y, float action) results)
        {
            if (DateTime.Now - lastInferenceTime < TimeSpan.FromMilliseconds(125))
            {
                results = lastInferenceValue;
                return;
            }

            OrtValue inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);

            var inputs = new Dictionary<string, OrtValue>
            {
                { "input", inputOrtValue }
            };

            using (var outputs = _session.Run(_runOptions, inputs, _session.OutputNames))
            {
                var outputToFeed = outputs.First().GetTensorDataAsSpan<float>();
                results = (outputToFeed[0], outputToFeed[1], outputToFeed[2]);
                Console.WriteLine(results);
            }

            lastInferenceTime = DateTime.Now;
            lastInferenceValue = results;

            inputOrtValue.Dispose(); // Important to free Tensor memory

        }
    }
}