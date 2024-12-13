using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Sleap;
using OpenCV.Net;
using Accord.Math.Optimization;

[Combinator]
[Description("Combines Identity from a Quad cam model with the full pose from a top camera model")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CombineTopAndQuadIds
{
    static double EucDist(Point2f point, Point2f other)
    {
        var dx = point.X - other.X;
        var dy = point.Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    public IObservable<PoseIdentityCollection> Process(IObservable<Tuple<PoseIdentityCollection, PoseCollection>> source)
    {
        return source.Select(value =>
        {
            var quadCamIdentities = value.Item1;
            var topCamPoses = value.Item2;
            var output = new PoseIdentityCollection(topCamPoses.Image, topCamPoses.Model);

                  // Return empty output if no poses or identities or if quadCamIdentities PoseIdentity is null
                if (topCamPoses.Count == 0 || quadCamIdentities.Count == 0 || quadCamIdentities == null)
            {
                Console.WriteLine("No poses or identities detected, returning empty output");
                return output;
            }

            foreach (var pose in topCamPoses)
            {
                var bestMatch = quadCamIdentities
                    .Where(identity => identity != null && !double.IsNaN(identity.Centroid.Position.X))
                    .OrderBy(identity => EucDist(pose["spine2"].Position, identity.Centroid.Position))
                    .FirstOrDefault();

                if (bestMatch != null)
                {
                    var unifiedPoseIdentity = new PoseIdentity(pose.Image, pose.Model)
                    {
                        Identity = bestMatch.Identity,
                        IdentityIndex = bestMatch.IdentityIndex,
                        IdentityScores = bestMatch.IdentityScores,
                        Confidence = bestMatch.Confidence,
                        Centroid = bestMatch.Centroid
                    };

                    foreach (var bodyPart in pose)
                    {
                        unifiedPoseIdentity.Add(bodyPart);
                    }

                    output.Add(unifiedPoseIdentity);
                }
            }

            return output;
        });
    }
}