using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;

[Combinator]
[Description("Filters and returns a PoseIdentityCollection containing the highest-confidence poses for each identity across the provided topCamCollection and quadCamCollections. Outputs a boolean indicating if the top camera collection was used as fallback for any identity.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class FindHighestConfidencePoseFallback
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

    public IObservable<Tuple<PoseIdentityCollection, bool>> Process(IObservable<Tuple<PoseIdentityCollection, IList<PoseIdentityCollection>>> source)
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

            // Combine all PoseIdentities from quad cams into a single list
            var allPoses = new List<PoseIdentity>();

            if (quadCamCollections != null)
            {
                foreach (var collection in quadCamCollections)
                {
                    if (collection != null)
                    {
                        allPoses.AddRange(collection);
                    }
                }
            }

            // Track if the topCamCollection was used as fallback
            bool usedFallback = false;

            // Group valid poses by identity
            var groupedByIdentity = allPoses
                .Where(pose => !string.IsNullOrEmpty(pose.Identity))
                .GroupBy(pose => pose.Identity)
                .ToDictionary(group => group.Key, group => group.OrderByDescending(pose => pose.Confidence).FirstOrDefault());

            foreach (var identity in model.ClassNames)
            {
                PoseIdentity highestConfidencePose = null;
                if (groupedByIdentity.TryGetValue(identity, out highestConfidencePose))
                {
                    outputCollection.Add(highestConfidencePose);
                }
                else
                {
                    // Fallback to top camera collection if no pose for this identity is found
                    usedFallback = true;

                    var topCamPose = topCamCollection.FirstOrDefault(p => p.Identity == identity);

                    if (topCamPose != null)
                    {
                        outputCollection.Add(topCamPose);
                    }
                    else
                    {
                        // Add an empty pose if no valid pose is found in top camera
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
            }

            return Tuple.Create(outputCollection, usedFallback);
        });
    }
}