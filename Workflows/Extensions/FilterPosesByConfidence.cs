using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("Filters and returns a PoseIdentityCollection containing the highest-confidence poses for each mouse collected from multiple cameras.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FilterPosesByConfidence
{   
    internal static BodyPart DefaultBodyPart(string name)
        {
            return new BodyPart
            {
                Name = name,
                Position = new Point2f(float.NaN, float.NaN),
                Confidence = float.NaN
            };
        }
    public IObservable<PoseIdentityCollection> Process(
        IObservable<Tuple<PoseIdentityCollection, PoseIdentityCollection, PoseIdentityCollection, PoseIdentityCollection>> source)
    {
        return source.Select(inputCollections =>
        {   var model = inputCollections.Item1.Model;
            var image = inputCollections.Item1.Image;
            // Create a new PoseIdentityCollection for the output
            var outputCollection = new PoseIdentityCollection(image, model);

            // Flatten all valid PoseIdentities from the collections into a single list
            var allPoses = new List<PoseIdentity>();
            if (inputCollections.Item1.Count > 0) allPoses.AddRange(inputCollections.Item1); // is if neccessary?
            if (inputCollections.Item2.Count > 0) allPoses.AddRange(inputCollections.Item2);
            if (inputCollections.Item3.Count > 0) allPoses.AddRange(inputCollections.Item3);
            if (inputCollections.Item4.Count > 0) allPoses.AddRange(inputCollections.Item4);

            // Group poses by their Identity string
            var groupedByIdentity = allPoses
                // .Where(pose => !string.IsNullOrEmpty(pose.Identity)) // Ensure identity is valid
                .GroupBy(pose => pose.Identity);

            // For each identity, find the pose with the highest confidence.
            foreach (var identityGroup in groupedByIdentity)
            {
                var highestConfidencePose = identityGroup
                    .OrderByDescending(pose => pose.Confidence)
                    .FirstOrDefault();

                // If not valid, create empty pose
                if (highestConfidencePose == null)
                {
                    highestConfidencePose = new PoseIdentity(image, model);
                    highestConfidencePose.Confidence = float.NaN;
                    highestConfidencePose.IdentityScores = new float[model.ClassNames.Count];
                    highestConfidencePose.Centroid = DefaultBodyPart(model.AnchorName);

                    foreach (var partName in model.PartNames)
                        {
                            highestConfidencePose.Add(DefaultBodyPart(partName));
                        }
                }
                outputCollection.Add(highestConfidencePose);

            }

            return outputCollection;
        });
    }
}