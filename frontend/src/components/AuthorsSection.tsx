import { Github, Linkedin, Mail, ExternalLink } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface Author {
  name: string;
  role: string;
  institution: string;
  bio: string;
  image: string;
  linkedin?: string;
  github?: string;
  email?: string;
  website?: string;
}

export function AuthorsSection() {
  // Mock author data - replace with actual team information
  const authors: Author[] = [
    {
      name: "Dr. Sarah Chen",
      role: "Lead AI Researcher", 
      institution: "Stanford Medical AI Lab",
      bio: "Specialist in medical imaging and deep learning with 8+ years in neuroimaging research.",
      image: "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300&h=300&fit=crop&crop=face",
      linkedin: "https://linkedin.com/in/sarahchen",
      github: "https://github.com/sarahchen",
      email: "sarah.chen@stanford.edu"
    },
    {
      name: "Dr. Michael Rodriguez",
      role: "Clinical Neurosurgeon",
      institution: "Johns Hopkins Hospital", 
      bio: "Board-certified neurosurgeon with expertise in brain tumor diagnosis and treatment.",
      image: "https://images.unsplash.com/photo-1612349317150-e413f6a5b16d?w=300&h=300&fit=crop&crop=face",
      linkedin: "https://linkedin.com/in/mrodriguez",
      email: "m.rodriguez@jhmi.edu",
      website: "https://hopkinsmedicine.org"
    },
    {
      name: "Dr. Priya Patel",
      role: "Machine Learning Engineer",
      institution: "MIT CSAIL",
      bio: "PhD in Computer Science focusing on Bayesian deep learning and uncertainty quantification.",
      image: "https://images.unsplash.com/photo-1594824122472-e2cc2dfd1e88?w=300&h=300&fit=crop&crop=face",
      linkedin: "https://linkedin.com/in/priyapatel", 
      github: "https://github.com/ppatel",
      email: "priya@mit.edu"
    },
    {
      name: "Dr. James Kim",
      role: "Medical Imaging Specialist",
      institution: "UCSF Radiology",
      bio: "Radiologist specializing in neuroimaging with focus on MRI sequence optimization.",
      image: "https://images.unsplash.com/photo-1582750433449-648ed127bb54?w=300&h=300&fit=crop&crop=face",
      linkedin: "https://linkedin.com/in/jameskim",
      email: "james.kim@ucsf.edu",
      website: "https://radiology.ucsf.edu"
    }
  ];

  return (
    <section id="authors" className="min-h-screen py-20 px-6 bg-background/20 backdrop-blur-[1px]">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-16 fade-in">
          <h2 className="text-medical-lg mb-6">Research Team</h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our interdisciplinary team combines expertise in artificial intelligence, 
            neurosurgery, radiology, and clinical medicine to advance brain tumor detection.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {authors.map((author, index) => (
            <Card key={index} className="card-medical p-6 text-center group">
              {/* Profile Image */}
              <div className="relative mb-6">
                <div className="w-32 h-32 mx-auto rounded-full overflow-hidden ring-4 ring-primary/20 group-hover:ring-primary/40 transition-all">
                  <img 
                    src={author.image}
                    alt={author.name}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
              </div>

              {/* Author Info */}
              <div className="space-y-3 mb-6">
                <h3 className="text-xl font-bold text-foreground">
                  {author.name}
                </h3>
                <div className="space-y-1">
                  <p className="text-primary font-semibold">
                    {author.role}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    {author.institution}
                  </p>
                </div>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {author.bio}
                </p>
              </div>

              {/* Social Links */}
              <div className="flex justify-center space-x-2">
                {author.linkedin && (
                  <Button 
                    variant="outline" 
                    size="icon"
                    asChild
                    className="hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600"
                  >
                    <a 
                      href={author.linkedin} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      aria-label={`${author.name}'s LinkedIn`}
                    >
                      <Linkedin className="w-4 h-4" />
                    </a>
                  </Button>
                )}
                
                {author.github && (
                  <Button 
                    variant="outline" 
                    size="icon"
                    asChild
                    className="hover:bg-gray-50 hover:border-gray-300 hover:text-gray-800"
                  >
                    <a 
                      href={author.github} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      aria-label={`${author.name}'s GitHub`}
                    >
                      <Github className="w-4 h-4" />
                    </a>
                  </Button>
                )}
                
                {author.email && (
                  <Button 
                    variant="outline" 
                    size="icon"
                    asChild
                    className="hover:bg-green-50 hover:border-green-300 hover:text-green-600"
                  >
                    <a 
                      href={`mailto:${author.email}`}
                      aria-label={`Email ${author.name}`}
                    >
                      <Mail className="w-4 h-4" />
                    </a>
                  </Button>
                )}
                
                {author.website && (
                  <Button 
                    variant="outline" 
                    size="icon"
                    asChild
                    className="hover:bg-purple-50 hover:border-purple-300 hover:text-purple-600"
                  >
                    <a 
                      href={author.website} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      aria-label={`${author.name}'s website`}
                    >
                      <ExternalLink className="w-4 h-4" />
                    </a>
                  </Button>
                )}
              </div>
            </Card>
          ))}
        </div>

        {/* Collaboration CTA */}
        <div className="mt-16 text-center">
          <Card className="card-medical p-8 max-w-3xl mx-auto">
            <h3 className="text-2xl font-bold mb-4">Research Collaboration</h3>
            <p className="text-muted-foreground mb-6">
              Interested in collaborating on medical AI research or integrating NeuroVision AI 
              into your clinical workflow? We welcome partnerships with healthcare institutions 
              and research organizations.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button className="btn-medical">
                <Mail className="w-4 h-4 mr-2" />
                Contact Research Team
              </Button>
              <Button variant="outline" className="border-primary text-primary hover:bg-primary/5">
                <ExternalLink className="w-4 h-4 mr-2" />
                View Publications
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}