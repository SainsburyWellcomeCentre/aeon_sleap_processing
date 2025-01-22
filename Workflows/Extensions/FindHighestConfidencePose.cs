using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("Filters and returns a PoseIdentityCollection containing the highest-confidence poses for each identity across the provided topCamCollection and quadCamCollections.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FindHighestConfidencePose
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

    public IObservable<PoseIdentityCollection> Process(IObservable<Tuple<PoseIdentityCollection, IList<PoseIdentityCollection>>> source)
    {
        return source.Select(input =>
        {
            var topCamCollection = input.Item1;
            var quadCamCollections = input.Item2;

            if (topCamCollection == null && (quadCamCollections == null || quadCamCollections.Count == 0))
            {
                throw new ArgumentException("Please provide a poseIdentityCollection from the top camera, and a list of poseIdentityCollections from the quad cameras");
            }

            // Initialize an empty output PoseIdentityCollection
            var model = topCamCollection.Model;
            var image = topCamCollection.Image;
            var outputCollection = new PoseIdentityCollection(image, model);

            // Combine all PoseIdentities into a single list
            var allPoses = new List<PoseIdentity>();
            if (topCamCollection != null) allPoses.AddRange(topCamCollection);
            if (quadCamCollections != null)
            {
                allPoses.AddRange(quadCamCollections.SelectMany(collection => collection ?? Enumerable.Empty<PoseIdentity>()));
            }

            // Group valid poses by identity
            var groupedByIdentity = allPoses
                .Where(pose => !string.IsNullOrEmpty(pose.Identity))
                .GroupBy(pose => pose.Identity);

            foreach (var identityGroup in groupedByIdentity)
            {
                // Get the pose with the highest confidence for each identity
                var highestConfidencePose = identityGroup
                    .OrderByDescending(pose => pose.Confidence)
                    .FirstOrDefault();

                if (highestConfidencePose != null)
                {
                    outputCollection.Add(highestConfidencePose);
                }
            }

            // If no valid poses are found, create empty poses for each expected identity
            if (outputCollection.Count == 0)
            {
                foreach (var identity in model.ClassNames)
                {
                    var emptyPose = new PoseIdentity(image, model)
                    {
                        Identity = identity,
                        Confidence = float.NaN,
                        IdentityScores = new float[model.ClassNames.Count],
                        Centroid = DefaultBodyPart(model.AnchorName)
                    };

                    foreach (var partName in model.PartNames)
                    {
                        emptyPose.Add(DefaultBodyPart(partName));
                    }

                    outputCollection.Add(emptyPose);
                }
            }
            return outputCollection;
        });
    }
}