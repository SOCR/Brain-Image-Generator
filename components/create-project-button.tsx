'use client';

import { PlusCircle } from 'lucide-react';
import { RainbowButton } from "@/components/ui/rainbow-button";
import { CreateProjectModal } from '@/components/create-project-modal';

export function CreateProjectButton() {
  const handleOpenModal = () => {
    document.dispatchEvent(new CustomEvent('open-create-project-modal'));
  };

  return (
    <>
      <RainbowButton onClick={handleOpenModal} size="lg" className="gap-2 shadow-lg">
        <PlusCircle className="h-4 w-4" />
        Create New Project
      </RainbowButton>
      <CreateProjectModal />
    </>
  );
}
